# FLOPs and parameter counting follow the NTIRE 2026 ESR rules:
# https://github.com/Amazingren/NTIRE2026_ESR

from __future__ import annotations

import ast
import contextlib
import csv
import gc
import hashlib
import importlib.util
import inspect
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis, flop_count_table


TRAINER_DIRECTORY = Path(r"nnUNetTrainer").expanduser().resolve()

OUTPUT_CSV = TRAINER_DIRECTORY / "parameters_gflops_fv.csv"

DUMMY_BATCH_SIZE = 1
DUMMY_INPUT_CHANNELS = 3
DUMMY_HEIGHT = 256
DUMMY_WIDTH = 256
DUMMY_OUTPUT_CHANNELS = 2
DUMMY_DTYPE = torch.float32

ARCHITECTURE_CLASS_NAME = ""
ENABLE_DEPLOY_CONVERSION = False

PRINT_FVCORE_TABLE = False
RANDOM_SEED = 12345


@dataclass(frozen=True)
class TrainerSpec:
    script_path: Path
    class_name: str


@dataclass(frozen=True)
class ResultRow:
    trainer_name: str
    parameters: int
    flops: int



def discover_trainers(directory: Path) -> list[TrainerSpec]:
    if not directory.exists():
        raise FileNotFoundError(
            f"TRAINER_DIRECTORY not in: {directory}"
        )
    if not directory.is_dir():
        raise NotADirectoryError(
            f"TRAINER_DIRECTORY not in: {directory}"
        )

    found: list[TrainerSpec] = []

    for script_path in sorted(directory.glob("*.py")):
        if script_path.name.startswith("."):
            continue

        try:
            source = script_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(script_path))
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            print(
                f"[SCAN SKIP] {script_path}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            continue

        for node in tree.body:
            if (
                isinstance(node, ast.ClassDef)
                and node.name.startswith("nnUNetTrainer_")
            ):
                found.append(
                    TrainerSpec(
                        script_path=script_path.resolve(),
                        class_name=node.name,
                    )
                )

    unique: list[TrainerSpec] = []
    seen: set[tuple[str, str]] = set()

    for trainer in found:
        key = (str(trainer.script_path), trainer.class_name)
        if key not in seen:
            unique.append(trainer)
            seen.add(key)

    return unique



def unique_module_name(script_path: Path) -> str:
    digest = hashlib.sha1(
        str(script_path).encode("utf-8")
    ).hexdigest()[:12]
    return f"_params_gflops_{script_path.stem}_{digest}"


def import_search_paths(script_path: Path) -> list[Path]:
    candidates = [
        script_path.parent,
        TRAINER_DIRECTORY,
        *TRAINER_DIRECTORY.parents,
    ]

    result: list[Path] = []
    seen: set[str] = set()

    for candidate in candidates:
        value = str(candidate)
        if value not in seen:
            result.append(candidate)
            seen.add(value)

    return result


def find_local_module_file(
    missing_module_name: str,
    script_path: Path,
) -> Path | None:
    prefix = "nnunetv2.training.nnUNetTrainer."

    if not missing_module_name.startswith(prefix):
        return None

    stem = missing_module_name.rsplit(".", 1)[-1]
    candidates = (
        script_path.parent / f"{stem}.py",
        TRAINER_DIRECTORY / f"{stem}.py",
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


def load_module_under_name(
    module_name: str,
    script_path: Path,
) -> ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name,
        script_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"{script_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def import_script(
    script_path: Path,
    module_cache: dict[Path, ModuleType],
) -> ModuleType:
    resolved = script_path.resolve()

    if resolved in module_cache:
        return module_cache[resolved]

    inserted_paths: list[str] = []

    for directory in import_search_paths(resolved):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)
            inserted_paths.append(value)

    module_name = unique_module_name(resolved)

    try:
        for _ in range(10):
            sys.modules.pop(module_name, None)

            try:
                module = load_module_under_name(
                    module_name,
                    resolved,
                )
                module_cache[resolved] = module
                return module

            except ModuleNotFoundError as error:
                sys.modules.pop(module_name, None)
                missing_name = error.name or ""
                local_file = find_local_module_file(
                    missing_name,
                    resolved,
                )

                if local_file is None:
                    raise

                load_module_under_name(
                    missing_name,
                    local_file,
                )

        raise ImportError(
            f"{resolved}"
        )

    finally:
        for value in inserted_paths:
            try:
                sys.path.remove(value)
            except ValueError:
                pass


def get_trainer_class(
    module: ModuleType,
    class_name: str,
) -> type:
    trainer_class = getattr(module, class_name, None)

    if trainer_class is None:
        raise AttributeError(
            f"{class_name}."
        )
    if not inspect.isclass(trainer_class):
        raise TypeError(
            f"{class_name} is not class"
        )
    if not callable(
        getattr(trainer_class, "build_network_architecture", None)
    ):
        raise AttributeError(
            f"{class_name} build_network_architecture()."
        )

    return trainer_class



def build_network(
    trainer_class: type,
    input_channels: int,
    output_channels: int,
    architecture_class_name: str,
    architecture_kwargs: Mapping[str, Any],
    architecture_imports: Sequence[str],
) -> nn.Module:
    builder = trainer_class.build_network_architecture

    keyword_arguments = {
        "architecture_class_name": architecture_class_name,
        "arch_init_kwargs": dict(architecture_kwargs),
        "arch_init_kwargs_req_import": list(architecture_imports),
        "num_input_channels": input_channels,
        "num_output_channels": output_channels,
        "enable_deep_supervision": False,
    }

    errors: list[str] = []

    try:
        model = builder(**keyword_arguments)
    except Exception as keyword_error:
        errors.append(
            f"keyword: {type(keyword_error).__name__}: {keyword_error}"
        )

        try:
            model = builder(
                architecture_class_name,
                dict(architecture_kwargs),
                list(architecture_imports),
                input_channels,
                output_channels,
                False,
            )
        except Exception as positional_error:
            errors.append(
                f"positional: {type(positional_error).__name__}: "
                f"{positional_error}"
            )
            raise RuntimeError(
                "build_network_architecture() failed "
                + " | ".join(errors)
            ) from positional_error

    if not isinstance(model, nn.Module):
        raise TypeError(
            "torch.nn.Module."
        )

    return model


def disable_deep_supervision(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "deep_supervision"):
            try:
                module.deep_supervision = False
            except Exception:
                pass


def switch_to_deploy_if_available(
    model: nn.Module,
    enabled: bool,
) -> int:
    if not enabled:
        return 0

    model.eval()

    root_method = getattr(model, "switch_to_deploy", None)
    if callable(root_method):
        with torch.no_grad():
            root_method()
        model.eval()
        return 1

    named_modules = list(model.named_modules())
    candidate_names = {
        name
        for name, module in named_modules
        if name
        and callable(getattr(module, "switch_to_deploy", None))
    }

    selected: list[nn.Module] = []

    for name, module in named_modules:
        if name not in candidate_names:
            continue

        parts = name.split(".")
        parent_names = {
            ".".join(parts[:index])
            for index in range(1, len(parts))
        }

        if parent_names.isdisjoint(candidate_names):
            selected.append(module)

    with torch.no_grad():
        for module in selected:
            module.switch_to_deploy()

    model.eval()
    return len(selected)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, Mapping):
        preferred_keys = (
            "logits",
            "prediction",
            "pred",
            "output",
            "segmentation",
        )

        for key in preferred_keys:
            if key in output:
                try:
                    return first_tensor(output[key])
                except (TypeError, ValueError):
                    pass

        for value in output.values():
            try:
                return first_tensor(value)
            except (TypeError, ValueError):
                continue

    if isinstance(output, (list, tuple)):
        for value in output:
            try:
                return first_tensor(value)
            except (TypeError, ValueError):
                continue

    raise TypeError(
        "tensor"
    )



def count_flops(
    model: nn.Module,
    dummy_input: torch.Tensor,
) -> int:

    analysis = FlopCountAnalysis(model, dummy_input)

    if PRINT_FVCORE_TABLE:
        print(flop_count_table(analysis))

    return int(analysis.total())


def create_dummy_input() -> torch.Tensor:
    return torch.randn(
        DUMMY_BATCH_SIZE,
        DUMMY_INPUT_CHANNELS,
        DUMMY_HEIGHT,
        DUMMY_WIDTH,
        dtype=DUMMY_DTYPE,
        device=torch.device("cpu"),
    )


def initialize_csv(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        csv.writer(csv_file).writerow(
            ("trainer_name", "parameters_m", "gflops")
        )


def append_csv_row(output_path: Path, row: ResultRow) -> None:
    with output_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        csv.writer(csv_file).writerow(
            (
                row.trainer_name,
                f"{row.parameters / (1000**2):.3f}",
                f"{row.flops / (1000**3):.3f}",
            )
        )
        csv_file.flush()
        os.fsync(csv_file.fileno())


def shorten_error(error: Exception, max_length: int = 700) -> str:
    message = (
        f"{type(error).__name__}: {error}"
    ).replace("\n", " ")

    if len(message) > max_length:
        return message[: max_length - 3] + "..."

    return message


def release_model(model: nn.Module | None) -> None:
    del model
    gc.collect()



def main() -> None:
    torch.manual_seed(RANDOM_SEED)

    trainer_specs = discover_trainers(TRAINER_DIRECTORY)
    if not trainer_specs:
        raise RuntimeError(
            f"nnUNetTrainer_*  {TRAINER_DIRECTORY}"
        )

    initialize_csv(OUTPUT_CSV)

    module_cache: dict[Path, ModuleType] = {}
    success_count = 0
    failure_count = 0


    for index, trainer_spec in enumerate(trainer_specs, start=1):
        print(
            f"[{index}/{len(trainer_specs)}] "
            f"{trainer_spec.class_name}",
            flush=True,
        )

        model: nn.Module | None = None

        try:
            hidden_output = io.StringIO()

            with contextlib.redirect_stdout(
                hidden_output
            ), contextlib.redirect_stderr(hidden_output):
                module = import_script(
                    trainer_spec.script_path,
                    module_cache,
                )
                trainer_class = get_trainer_class(
                    module,
                    trainer_spec.class_name,
                )
                model = build_network(
                    trainer_class=trainer_class,
                    input_channels=DUMMY_INPUT_CHANNELS,
                    output_channels=DUMMY_OUTPUT_CHANNELS,
                    architecture_class_name=ARCHITECTURE_CLASS_NAME,
                    architecture_kwargs={},
                    architecture_imports=[],
                )

            disable_deep_supervision(model)
            model.eval()

            deploy_calls = switch_to_deploy_if_available(
                model,
                enabled=ENABLE_DEPLOY_CONVERSION,
            )
            model.eval()

            parameters = count_parameters(model)
            dummy_input = create_dummy_input()
            flops = count_flops(model, dummy_input)

            row = ResultRow(
                trainer_name=trainer_spec.class_name,
                parameters=parameters,
                flops=flops,
            )
            append_csv_row(OUTPUT_CSV, row)
            success_count += 1

            print(
                "    OK | "
                f"params={parameters / (1000**2):.3f} M | "
                f"GFLOPs={flops / (1000**3):.3f} | "
                f"deploy_calls={deploy_calls}",
                flush=True,
            )

            del dummy_input

        except KeyboardInterrupt:
            print(
                f"{OUTPUT_CSV}",
                file=sys.stderr,
                flush=True,
            )
            raise

        except Exception as error:
            failure_count += 1
            print(
                "    FAILED | "
                f"file={trainer_spec.script_path} | "
                f"{shorten_error(error)}",
                flush=True,
            )

        finally:
            release_model(model)



if __name__ == "__main__":
    main()
