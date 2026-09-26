from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import gc
import hashlib
import importlib.util
import inspect
import io
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import torch
from torch import nn



TRAINER_DIRECTORY = Path(
    r"/path/to/nnUNetTrainers"
).expanduser().resolve()

OUTPUT_CSV = TRAINER_DIRECTORY / "benchmark_cpu_gpu_vram.csv"

DUMMY_BATCH_SIZE = 1
DUMMY_INPUT_CHANNELS = 3
DUMMY_HEIGHT = 256
DUMMY_WIDTH = 256
DUMMY_OUTPUT_CHANNELS = 2
DUMMY_DTYPE = torch.float32


@dataclass(frozen=True)
class TrainerSpec:
    script_path: Path
    class_name: str


@dataclass(frozen=True)
class BenchmarkRow:
    trainer_name: str
    total_parameters: int
    inference_parameters: int
    mean_cpu_latency_ms: float
    mean_gpu_latency_ms: float | None
    peak_gpu_vram_allocated_mb: float | None



def discover_trainers(directory: Path) -> list[TrainerSpec]:
    if not directory.exists():
        raise FileNotFoundError(
            f"{directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"TRAINER_DIRECTORY is not: {directory}"
        )

    found: list[TrainerSpec] = []

    for script_path in sorted(directory.rglob("*.py")):
        if "__pycache__" in script_path.parts:
            continue
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

    for trainer_spec in found:
        key = (
            str(trainer_spec.script_path),
            trainer_spec.class_name,
        )
        if key not in seen:
            unique.append(trainer_spec)
            seen.add(key)

    return unique



def unique_module_name(script_path: Path) -> str:
    digest = hashlib.sha1(
        str(script_path).encode("utf-8")
    ).hexdigest()[:12]

    return (
        f"_trainer_benchmark_{script_path.stem}_{digest}"
    )



def import_search_paths(script_path: Path) -> list[Path]:
    candidates = [
        script_path.parent,
        TRAINER_DIRECTORY,
        *TRAINER_DIRECTORY.parents,
    ]

    result: list[Path] = []
    seen: set[str] = set()

    for candidate in candidates:
        candidate_string = str(candidate)
        if candidate_string not in seen:
            result.append(candidate)
            seen.add(candidate_string)

    return result



def import_script(
    script_path: Path,
    module_cache: dict[Path, ModuleType],
) -> ModuleType:
    resolved = script_path.resolve()

    if resolved in module_cache:
        return module_cache[resolved]

    module_name = unique_module_name(resolved)
    import_spec = importlib.util.spec_from_file_location(
        module_name,
        resolved,
    )

    if import_spec is None or import_spec.loader is None:
        raise ImportError(
            f" {resolved}"
        )

    module = importlib.util.module_from_spec(import_spec)

    inserted_paths: list[str] = []

    for directory in import_search_paths(resolved):
        directory_string = str(directory)
        if directory_string not in sys.path:
            sys.path.insert(0, directory_string)
            inserted_paths.append(directory_string)

    try:
        sys.modules[module_name] = module
        import_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        for directory_string in inserted_paths:
            try:
                sys.path.remove(directory_string)
            except ValueError:
                pass

    module_cache[resolved] = module
    return module



def get_trainer_class(
    module: ModuleType,
    class_name: str,
) -> type:
    trainer_class = getattr(module, class_name, None)

    if trainer_class is None:
        raise AttributeError(
            f"class{class_name} is not present"
        )

    if not inspect.isclass(trainer_class):
        raise TypeError(
            f"{class_name} is not a class "
        )

    builder = getattr(
        trainer_class,
        "build_network_architecture",
        None,
    )

    if not callable(builder):
        raise AttributeError(
            f"{class_name} does not have build_network_architecture()."
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

    builder = getattr(
        trainer_class,
        "build_network_architecture",
    )

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
            "keyword call: "
            f"{type(keyword_error).__name__}: {keyword_error}"
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
                "positional call: "
                f"{type(positional_error).__name__}: "
                f"{positional_error}"
            )

            raise RuntimeError(
                "build_network_architecture() a eșuat: "
                + " | ".join(errors)
            ) from positional_error

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"{type(model).__name__}, nu torch.nn.Module."
        )

    return model



def disable_deep_supervision(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "deep_supervision"):
            try:
                setattr(module, "deep_supervision", False)
            except Exception:
                pass



def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )



def switch_to_deploy_if_available(
    model: nn.Module,
    enabled: bool,
) -> tuple[bool, int]:

    if not enabled:
        return False, 0

    model.eval()

    root_method = getattr(model, "switch_to_deploy", None)

    if callable(root_method):
        with torch.inference_mode():
            root_method()
        model.eval()
        return True, 1

    named_modules = list(model.named_modules())

    candidate_names = {
        name
        for name, module in named_modules
        if name
        and callable(
            getattr(module, "switch_to_deploy", None)
        )
    }


    selected: list[tuple[str, nn.Module]] = []

    for name, module in named_modules:
        if name not in candidate_names:
            continue

        parts = name.split(".")
        parent_names = {
            ".".join(parts[:index])
            for index in range(1, len(parts))
        }

        if parent_names.isdisjoint(candidate_names):
            selected.append((name, module))

    converted_count = 0

    with torch.inference_mode():
        for _, module in selected:
            method = getattr(module, "switch_to_deploy", None)
            if callable(method):
                method()
                converted_count += 1

    model.eval()
    return converted_count > 0, converted_count




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
        f"Tip rezultat: {type(output).__name__}"
    )



@torch.inference_mode()
def warm_up(
    model: nn.Module,
    dummy_input: torch.Tensor,
    iterations: int,
) -> None:
    for _ in range(iterations):
        output = model(dummy_input)
        _ = first_tensor(output).shape


@torch.inference_mode()
def measure_mean_latency_ms(
    model: nn.Module,
    dummy_input: torch.Tensor,
    iterations: int,
) -> float:
    measurements_ms: list[float] = []

    gc_was_enabled = gc.isenabled()
    gc.collect()

    if gc_was_enabled:
        gc.disable()

    try:
        output: Any = None

        for _ in range(iterations):
            start_ns = time.perf_counter_ns()
            output = model(dummy_input)
            end_ns = time.perf_counter_ns()

            measurements_ms.append(
                (end_ns - start_ns) / 1_000_000.0
            )

        _ = first_tensor(output).shape
    finally:
        if gc_was_enabled:
            gc.enable()

    return statistics.fmean(measurements_ms)


@torch.inference_mode()
def measure_mean_gpu_latency_ms(
    model: nn.Module,
    dummy_input: torch.Tensor,
    iterations: int,
) -> float:
    measurements_ms: list[float] = []

    start_event = torch.cuda.Event(
        enable_timing=True
    )
    end_event = torch.cuda.Event(
        enable_timing=True
    )

    output: Any = None

    for _ in range(iterations):
        start_event.record()
        output = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize()

        measurements_ms.append(
            start_event.elapsed_time(end_event)
        )

    _ = first_tensor(output).shape

    return statistics.fmean(measurements_ms)


def create_dummy_input(
    memory_format: str,
) -> torch.Tensor:
    dummy_input = torch.randn(
        DUMMY_BATCH_SIZE,
        DUMMY_INPUT_CHANNELS,
        DUMMY_HEIGHT,
        DUMMY_WIDTH,
        dtype=DUMMY_DTYPE,
        device="cpu",
    )

    if memory_format == "channels_last":
        dummy_input = dummy_input.contiguous(
            memory_format=torch.channels_last
        )

    return dummy_input



def validate_forward(
    model: nn.Module,
    dummy_input: torch.Tensor,
) -> tuple[int, ...]:
    with torch.inference_mode():
        output = first_tensor(model(dummy_input))

    if output.ndim < 3:
        raise RuntimeError(
            "Output invalid: "
            f"shape={tuple(output.shape)}"
        )

    if output.shape[0] != DUMMY_BATCH_SIZE:
        raise RuntimeError(
            "Batch output: "
            f"shape={tuple(output.shape)}"
        )

    return tuple(output.shape)



CSV_COLUMNS = (
    "trainer_name",
    "total_parameters",
    "inference_parameters",
    "mean_cpu_latency_ms",
    "mean_gpu_latency_ms",
    "peak_gpu_vram_allocated_mb",
)



def initialize_csv(output_path: Path) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_COLUMNS)



def append_csv_row(
    output_path: Path,
    row: BenchmarkRow,
) -> None:
    with output_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            (
                row.trainer_name,
                row.total_parameters,
                row.inference_parameters,
                f"{row.mean_cpu_latency_ms:.6f}",
                (
                    f"{row.mean_gpu_latency_ms:.6f}"
                    if row.mean_gpu_latency_ms is not None
                    else ""
                ),
                (
                    f"{row.peak_gpu_vram_allocated_mb:.6f}"
                    if row.peak_gpu_vram_allocated_mb is not None
                    else ""
                ),
            )
        )
        csv_file.flush()
        os.fsync(csv_file.fileno())



def set_cpu_affinity(cpu_list: str | None) -> str:
    if not cpu_list:
        return "unspecified"

    if not hasattr(os, "sched_setaffinity"):
        return "not supported"

    cpus = {
        int(value.strip())
        for value in cpu_list.split(",")
        if value.strip()
    }

    if not cpus:
        raise ValueError(
            "--cpu-affinity "
        )

    os.sched_setaffinity(0, cpus)

    return ",".join(
        str(cpu)
        for cpu in sorted(cpus)
    )



def progress_text(
    current: int,
    total: int,
    trainer_name: str,
) -> str:
    percentage = (
        100.0 * current / total
        if total > 0
        else 100.0
    )

    return (
        f"[{current}/{total} | {percentage:6.2f}%] "
        f"{trainer_name}"
    )



def shorten_error(
    error: Exception,
    max_length: int = 500,
) -> str:
    message = (
        f"{type(error).__name__}: {error}"
    ).replace("\n", " ")

    if len(message) > max_length:
        return message[: max_length - 3] + "..."

    return message



def release_model(model: nn.Module | None) -> None:
    if model is not None:
        try:
            model.cpu()
        except Exception:
            pass

    del model
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--cpu-affinity",
        default=None,
    )
    parser.add_argument(
        "--memory-format",
        choices=("contiguous", "channels_last"),
        default="contiguous",
    )
    parser.add_argument(
        "--disable-mkldnn",
        action="store_true",
    )
    parser.add_argument(
        "--no-deploy-conversion",
        action="store_true",
    )
    parser.add_argument(
        "--architecture-class-name",
        default="",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs has to be positive.")
    if args.warmup < 0:
        parser.error("--warmup has to be positive.")
    if args.threads <= 0:
        parser.error("--threads has to be positive.")

    affinity_description = set_cpu_affinity(
        args.cpu_affinity
    )

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    torch.backends.mkldnn.enabled = (
        not args.disable_mkldnn
    )

    trainer_specs = discover_trainers(
        TRAINER_DIRECTORY
    )

    if not trainer_specs:
        raise RuntimeError(
            "no nnUNetTrainer_* classes in "
            f" {TRAINER_DIRECTORY}"
        )

    initialize_csv(OUTPUT_CSV)

    total_trainers = len(trainer_specs)
    success_count = 0
    failure_count = 0
    module_cache: dict[Path, ModuleType] = {}


    for index, trainer_spec in enumerate(
        trainer_specs,
        start=1,
    ):
        print(
            progress_text(
                index,
                total_trainers,
                trainer_spec.class_name,
            ),
            flush=True,
        )

        model: nn.Module | None = None
        dummy_input: torch.Tensor | None = None
        dummy_input_gpu: torch.Tensor | None = None

        try:
            captured_output = io.StringIO()

            with contextlib.redirect_stdout(
                captured_output
            ), contextlib.redirect_stderr(
                captured_output
            ):
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
                    architecture_class_name=(
                        args.architecture_class_name
                    ),
                    architecture_kwargs={},
                    architecture_imports=[],
                )

            disable_deep_supervision(model)
            model.eval()

            total_parameters = count_parameters(model)

            deploy_applied, deploy_calls = (
                switch_to_deploy_if_available(
                    model,
                    enabled=(
                        not args.no_deploy_conversion
                    ),
                )
            )

            model.eval()

            inference_parameters = count_parameters(model)

            if args.memory_format == "channels_last":
                model = model.to(
                    memory_format=torch.channels_last
                )

            dummy_input = create_dummy_input(
                args.memory_format
            )

            output_shape = validate_forward(
                model,
                dummy_input,
            )

            warm_up(
                model,
                dummy_input,
                args.warmup,
            )

            mean_latency_ms = measure_mean_latency_ms(
                model,
                dummy_input,
                args.runs,
            )

            mean_gpu_latency_ms: float | None = None
            peak_gpu_vram_allocated_mb: float | None = None

            if torch.cuda.is_available():
                gpu_device = torch.device("cuda")
                model = model.to(gpu_device)
                dummy_input_gpu = dummy_input.to(gpu_device)

                validate_forward(
                    model,
                    dummy_input_gpu,
                )

                warm_up(
                    model,
                    dummy_input_gpu,
                    args.warmup,
                )
                torch.cuda.synchronize()


                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(gpu_device)

                mean_gpu_latency_ms = (
                    measure_mean_gpu_latency_ms(
                        model,
                        dummy_input_gpu,
                        args.runs,
                    )
                )

                torch.cuda.synchronize()
                peak_gpu_vram_allocated_mb = (
                    torch.cuda.max_memory_allocated(gpu_device)
                    / (1024**2)
                )

                dummy_input_gpu = None
                model = model.cpu()
                torch.cuda.empty_cache()

            benchmark_row = BenchmarkRow(
                trainer_name=trainer_spec.class_name,
                total_parameters=total_parameters,
                inference_parameters=inference_parameters,
                mean_cpu_latency_ms=mean_latency_ms,
                mean_gpu_latency_ms=mean_gpu_latency_ms,
                peak_gpu_vram_allocated_mb=(
                    peak_gpu_vram_allocated_mb
                ),
            )

            append_csv_row(
                OUTPUT_CSV,
                benchmark_row,
            )

            success_count += 1

            graph_label = (
                "deploy"
                if deploy_applied
                else "native"
            )

            removed_parameters = (
                total_parameters - inference_parameters
            )

            gpu_text = (
                f"{mean_gpu_latency_ms:.6f} ms"
                if mean_gpu_latency_ms is not None
                else "N/A"
            )
            vram_text = (
                f"{peak_gpu_vram_allocated_mb:.2f} MiB allocated"
                if peak_gpu_vram_allocated_mb is not None
                else "N/A"
            )

            print(
                "    OK | "
                f"trainer={trainer_spec.class_name} | "
                f"params_total={total_parameters:,} | "
                f"params_inference={inference_parameters:,} | "
                f"removed={removed_parameters:,} | "
                f"mean_cpu={mean_latency_ms:.6f} ms | "
                f"mean_gpu={gpu_text} | "
                f"peak_vram={vram_text} | "
                f"output={output_shape} | "
                f"graph={graph_label} | "
                f"deploy_calls={deploy_calls}",
                flush=True,
            )

            dummy_input = None

        except KeyboardInterrupt:
            print(
                f" {OUTPUT_CSV}",
                file=sys.stderr,
                flush=True,
            )
            raise

        except Exception as error:
            failure_count += 1

            print(
                "    FAILED | "
                f"trainer={trainer_spec.class_name} | "
                f"file={trainer_spec.script_path} | "
                f"{shorten_error(error)}",
                flush=True,
            )

        finally:
            dummy_input_gpu = None
            dummy_input = None
            release_model(model)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()



if __name__ == "__main__":
    main()

