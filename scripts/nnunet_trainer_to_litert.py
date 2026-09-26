# python nnunet_trainer_to_litert.py /path/to/nnUNetTrainer_LightMIS.py

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import torch
from torch import nn


BATCH_SIZE = 1
INPUT_CHANNELS = 3
HEIGHT = 256
WIDTH = 256
OUTPUT_CHANNELS = 2
INPUT_DTYPE = torch.float32

OUTPUT_ROOT = Path("litert_exports").resolve()
SAVE_PT2 = True
TORCH_SEED = 2026



def detect_trainer_class_name(script_path: Path) -> str:
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))

    names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name.startswith("nnUNetTrainer_")
    ]

    if not names:
        raise RuntimeError(
            f"No class beginning with 'nnUNetTrainer_' was found in:\n"
            f"{script_path}"
        )

    if len(names) > 1:
        raise RuntimeError(
            "More than one nnUNetTrainer_* class is defined in the file:\n"
            + "\n".join(f"  - {name}" for name in names)
        )

    return names[0]



def unique_module_name(script_path: Path) -> str:
    digest = hashlib.sha1(
        str(script_path.resolve()).encode("utf-8")
    ).hexdigest()[:12]

    return f"_litert_trainer_{script_path.stem}_{digest}"


def candidate_import_paths(script_path: Path) -> list[Path]:
    script_path = script_path.resolve()

    candidates = [
        script_path.parent,
        *script_path.parent.parents,
    ]

    result: list[Path] = []
    seen: set[str] = set()

    for path in candidates:
        value = str(path)
        if value not in seen:
            result.append(path)
            seen.add(value)

    return result


def find_local_nnunet_module(
    missing_module_name: str,
    trainer_path: Path,
) -> Path | None:

    prefix = "nnunetv2.training.nnUNetTrainer."

    if not missing_module_name.startswith(prefix):
        return None

    stem = missing_module_name.rsplit(".", 1)[-1]

    candidates = [
        trainer_path.parent / f"{stem}.py",
    ]

    for parent in list(trainer_path.parent.parents)[:4]:
        candidates.append(parent / f"{stem}.py")
        candidates.append(
            parent / "nnUNetTrainer" / f"{stem}.py"
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
            f"Could not create an import spec for {script_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def import_trainer_script(script_path: Path) -> ModuleType:
    script_path = script_path.resolve()
    module_name = unique_module_name(script_path)

    inserted_paths: list[str] = []

    for path in candidate_import_paths(script_path):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
            inserted_paths.append(value)

    try:
        for _ in range(12):
            sys.modules.pop(module_name, None)

            try:
                return load_module_under_name(
                    module_name,
                    script_path,
                )

            except ModuleNotFoundError as exc:
                sys.modules.pop(module_name, None)

                missing_name = exc.name or ""
                local_file = find_local_nnunet_module(
                    missing_name,
                    script_path,
                )

                if local_file is None:
                    raise

                load_module_under_name(
                    missing_name,
                    local_file,
                )

        raise ImportError(
            "Too many recursive local imports while loading trainer."
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
    trainer_class = getattr(
        module,
        class_name,
        None,
    )

    if trainer_class is None:
        raise AttributeError(
            f"Class {class_name!r} was detected in source but was not present "
            "after importing the module."
        )

    if not inspect.isclass(trainer_class):
        raise TypeError(
            f"{class_name!r} exists but is not a class."
        )

    builder = getattr(
        trainer_class,
        "build_network_architecture",
        None,
    )

    if not callable(builder):
        raise AttributeError(
            f"{class_name!r} does not expose build_network_architecture()."
        )

    return trainer_class



def build_network(trainer_class: type) -> nn.Module:
    builder = trainer_class.build_network_architecture

    keyword_arguments = {
        "architecture_class_name": "",
        "arch_init_kwargs": {},
        "arch_init_kwargs_req_import": [],
        "num_input_channels": INPUT_CHANNELS,
        "num_output_channels": OUTPUT_CHANNELS,
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
                "",
                {},
                [],
                INPUT_CHANNELS,
                OUTPUT_CHANNELS,
                False,
            )

        except Exception as positional_error:
            errors.append(
                "positional call: "
                f"{type(positional_error).__name__}: "
                f"{positional_error}"
            )

            raise RuntimeError(
                "Could not build the network with the standard nnU-Net custom "
                "trainer interface.\n"
                + "\n".join(errors)
            ) from positional_error

    if not isinstance(model, nn.Module):
        raise TypeError(
            "build_network_architecture() did not return torch.nn.Module."
        )

    return model.cpu().eval()


def primary_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)) and output:
        first = output[0]

        if isinstance(first, torch.Tensor):
            return first

    raise TypeError(
        "The model output could not be normalized to a single Tensor. "
        f"Received: {type(output)}"
    )


class TensorOutputWrapper(nn.Module):

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x)

        if isinstance(output, torch.Tensor):
            return output

        return output[0]


def extract_state_dict(
    checkpoint: Any,
) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in (
            "network_weights",
            "state_dict",
            "model_state_dict",
            "model",
        ):
            value = checkpoint.get(key)

            if isinstance(value, Mapping):
                return value

        if checkpoint and all(
            isinstance(key, str)
            for key in checkpoint.keys()
        ):
            values = list(checkpoint.values())

            if values and all(
                isinstance(value, torch.Tensor)
                for value in values
            ):
                return checkpoint

    raise ValueError(
        "Could not identify a model state_dict in the checkpoint."
    )


def clean_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    prefixes = (
        "module.",
        "_orig_mod.",
        "network.",
    )

    result: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        clean = key

        changed = True
        while changed:
            changed = False

            for prefix in prefixes:
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
                    changed = True

        result[clean] = value

    return result


def load_checkpoint_if_requested(
    model: nn.Module,
    checkpoint_path: Path | None,
) -> str:
    if checkpoint_path is None:
        return "none (random initialized weights)"

    checkpoint_path = checkpoint_path.expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = clean_state_dict_keys(
        extract_state_dict(checkpoint)
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint is not an exact match for the built network.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )

    return str(checkpoint_path)


def _is_adaptive_global_max(module: nn.Module) -> bool:
    if not isinstance(module, nn.AdaptiveMaxPool2d):
        return False

    output_size = module.output_size

    if output_size == 1:
        return True

    if isinstance(output_size, (tuple, list)):
        return tuple(output_size) == (1, 1)

    return False


def _get_parent_module(
    root: nn.Module,
    qualified_name: str,
) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root

    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    return parent, parts[-1]


def _set_child_module(
    root: nn.Module,
    qualified_name: str,
    replacement: nn.Module,
) -> None:
    parent, child_name = _get_parent_module(
        root,
        qualified_name,
    )

    if child_name.isdigit():
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def auto_replace_adaptive_global_max(
    model: nn.Module,
    example_input: torch.Tensor,
) -> list[str]:

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name and _is_adaptive_global_max(module)
    ]

    if not targets:
        return []

    observed_shapes: dict[str, tuple[int, int]] = {}
    handles = []

    for name, module in targets:
        def make_hook(module_name: str):
            def hook(
                _module: nn.Module,
                inputs: tuple[Any, ...],
            ) -> None:
                if not inputs:
                    raise RuntimeError(
                        f"No input captured for {module_name}"
                    )

                tensor = inputs[0]

                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"AdaptiveMaxPool2d input for {module_name} "
                        f"is not a Tensor: {type(tensor)}"
                    )

                if tensor.ndim != 4:
                    raise RuntimeError(
                        f"Expected NCHW tensor at {module_name}, "
                        f"got shape {tuple(tensor.shape)}"
                    )

                observed_shapes[module_name] = (
                    int(tensor.shape[-2]),
                    int(tensor.shape[-1]),
                )

            return hook

        handles.append(
            module.register_forward_pre_hook(
                make_hook(name)
            )
        )

    try:
        with torch.inference_mode():
            _ = model(example_input)
    finally:
        for handle in handles:
            handle.remove()

    changes: list[str] = []

    for name, _module in targets:
        if name not in observed_shapes:
            raise RuntimeError(
                f"AdaptiveMaxPool2d module {name!r} was not executed "
                "during the fixed-shape calibration forward."
            )

        height, width = observed_shapes[name]

        if height <= 0 or width <= 0:
            raise RuntimeError(
                f"Invalid observed shape for {name}: "
                f"{height}x{width}"
            )

        replacement = nn.MaxPool2d(
            kernel_size=(height, width),
            stride=(height, width),
            padding=0,
            dilation=1,
            ceil_mode=False,
        )

        _set_child_module(
            model,
            name,
            replacement,
        )

        changes.append(
            f"{name}: AdaptiveMaxPool2d(1) -> "
            f"MaxPool2d(kernel={height}x{width}, stride={height}x{width})"
        )

    return changes



class ExactDepthwiseConvTranspose2dAsExpandShuffle(nn.Module):

    def __init__(self, source: nn.ConvTranspose2d):
        super().__init__()

        if not isinstance(source, nn.ConvTranspose2d):
            raise TypeError("source must be nn.ConvTranspose2d")

        k_h, k_w = source.kernel_size
        s_h, s_w = source.stride
        p_h, p_w = source.padding
        op_h, op_w = source.output_padding
        d_h, d_w = source.dilation

        if not (
            source.in_channels == source.out_channels == source.groups
            and k_h == k_w
            and s_h == s_w
            and k_h == s_h
            and p_h == p_w == 0
            and op_h == op_w == 0
            and d_h == d_w == 1
        ):
            raise ValueError(
                "Exact replacement supports only depthwise ConvTranspose2d "
                "with in=out=groups, square kernel=stride, padding=0, "
                "output_padding=0, dilation=1."
            )

        self.channels = int(source.in_channels)
        self.scale = int(k_h)

        self.expand = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels * self.scale * self.scale,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.channels,
            bias=source.bias is not None,
        )

        # ConvTranspose2d weight layout:
        #   [in_channels, out_channels/groups, kH, kW]
        # For depthwise out_channels/groups == 1.
        #
        # Grouped 1x1 Conv2d output ordering is:
        #   channel c -> r*r consecutive sub-pixel channels.
        #
        # Map [c, 0, i, j] -> [c*r*r + i*r + j, 0, 0, 0].
        with torch.no_grad():
            mapped_weight = (
                source.weight.detach()
                .reshape(
                    self.channels * self.scale * self.scale,
                    1,
                    1,
                    1,
                )
                .clone()
            )
            self.expand.weight.copy_(mapped_weight)

            if source.bias is not None:
                repeated_bias = (
                    source.bias.detach()
                    .repeat_interleave(
                        self.scale * self.scale
                    )
                    .clone()
                )
                self.expand.bias.copy_(repeated_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)

        batch = y.shape[0]
        height = y.shape[2]
        width = y.shape[3]
        r = self.scale
        c = self.channels

        # [B, C*r*r, H, W]
        # -> [B, C, r, r, H, W]
        # -> [B, C, H, r, W, r]
        # -> [B, C, H*r, W*r]
        y = y.reshape(
            batch,
            c,
            r,
            r,
            height,
            width,
        )
        y = y.permute(
            0, 1, 4, 2, 5, 3
        ).contiguous()
        y = y.reshape(
            batch,
            c,
            height * r,
            width * r,
        )

        return y


def auto_replace_supported_depthwise_convtranspose(
    model: nn.Module,
) -> list[str]:

    targets: list[tuple[str, nn.ConvTranspose2d]] = []

    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.ConvTranspose2d):
            continue

        k_h, k_w = module.kernel_size
        s_h, s_w = module.stride
        p_h, p_w = module.padding
        op_h, op_w = module.output_padding
        d_h, d_w = module.dilation

        supported = (
            module.in_channels
            == module.out_channels
            == module.groups
            and k_h == k_w
            and s_h == s_w
            and k_h == s_h
            and p_h == p_w == 0
            and op_h == op_w == 0
            and d_h == d_w == 1
        )

        if supported:
            targets.append((name, module))

    changes: list[str] = []

    for name, module in targets:
        replacement = (
            ExactDepthwiseConvTranspose2dAsExpandShuffle(
                module
            )
        )

        _set_child_module(
            model,
            name,
            replacement,
        )

        changes.append(
            f"{name}: depthwise ConvTranspose2d("
            f"C={module.in_channels}, "
            f"kernel={module.kernel_size[0]}, "
            f"stride={module.stride[0]}, "
            f"groups={module.groups}) -> "
            "exact grouped 1x1 Conv2d + reshape/permute"
        )

    return changes



class BroadcastSafeChannelAttentionBridge(nn.Module):
    """
        [B, C, 1, 1] -> [B, C, H, W]
    """

    def __init__(self, source: nn.Module):
        super().__init__()

        self.split_att = source.split_att
        self.avgpool = source.avgpool
        self.get_all_att = source.get_all_att
        self.att1 = source.att1
        self.att2 = source.att2
        self.att3 = source.att3
        self.att4 = source.att4
        self.att5 = source.att5
        self.sigmoid = source.sigmoid

    def forward(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        t3: torch.Tensor,
        t4: torch.Tensor,
        t5: torch.Tensor,
    ):
        att = torch.cat(
            (
                self.avgpool(t1),
                self.avgpool(t2),
                self.avgpool(t3),
                self.avgpool(t4),
                self.avgpool(t5),
            ),
            dim=1,
        )

        att = self.get_all_att(
            att.squeeze(-1).transpose(-1, -2)
        )

        if self.split_att != "fc":
            att = att.transpose(-1, -2)

        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))

        if self.split_att == "fc":
            # Original code:
            #   transpose -> unsqueeze -> expand_as(t_i)
            #
            # We intentionally stop at [B,C,1,1]. Multiplication with
            # [B,C,H,W] is exactly the same by broadcasting.
            att1 = att1.transpose(-1, -2).unsqueeze(-1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1)
            att3 = att3.transpose(-1, -2).unsqueeze(-1)
            att4 = att4.transpose(-1, -2).unsqueeze(-1)
            att5 = att5.transpose(-1, -2).unsqueeze(-1)
        else:
            att1 = att1.unsqueeze(-1)
            att2 = att2.unsqueeze(-1)
            att3 = att3.unsqueeze(-1)
            att4 = att4.unsqueeze(-1)
            att5 = att5.unsqueeze(-1)

        return att1, att2, att3, att4, att5


def _looks_like_expand_channel_attention_bridge(
    module: nn.Module,
) -> bool:

    required = (
        "split_att",
        "avgpool",
        "get_all_att",
        "att1",
        "att2",
        "att3",
        "att4",
        "att5",
        "sigmoid",
    )

    return all(
        hasattr(module, name)
        for name in required
    )


def auto_replace_expand_channel_attention_bridges(
    model: nn.Module,
) -> list[str]:
    targets: list[tuple[str, nn.Module]] = []

    for name, module in model.named_modules():
        if not name:
            continue

        if isinstance(
            module,
            BroadcastSafeChannelAttentionBridge,
        ):
            continue

        if _looks_like_expand_channel_attention_bridge(
            module
        ):
            targets.append(
                (name, module)
            )

    changes: list[str] = []

    targets.sort(
        key=lambda item: item[0].count("."),
        reverse=True,
    )

    for name, module in targets:
        replacement = (
            BroadcastSafeChannelAttentionBridge(
                module
            )
        )

        _set_child_module(
            model,
            name,
            replacement,
        )

        changes.append(
            f"{name}: explicit expand_as channel masks -> "
            "compact [B,C,1,1] masks with standard broadcasting"
        )

    return changes


def apply_generic_litert_compatibility(
    model: nn.Module,
    example_input: torch.Tensor,
) -> tuple[list[str], float]:

    model.eval()

    with torch.inference_mode():
        before = primary_output(
            model(example_input)
        ).detach().cpu()

    changes: list[str] = []

    changes.extend(
        auto_replace_adaptive_global_max(
            model,
            example_input,
        )
    )

    changes.extend(
        auto_replace_supported_depthwise_convtranspose(
            model,
        )
    )

    changes.extend(
        auto_replace_expand_channel_attention_bridges(
            model,
        )
    )

    if not changes:
        return [], 0.0

    with torch.inference_mode():
        after = primary_output(
            model(example_input)
        ).detach().cpu()

    if before.shape != after.shape:
        raise RuntimeError(
            "LiteRT compatibility rewrite changed output shape: "
            f"{tuple(before.shape)} -> {tuple(after.shape)}"
        )

    max_abs_diff = float(
        (before - after)
        .abs()
        .max()
        .item()
    )

    if max_abs_diff > 1e-5:
        raise RuntimeError(
            "LiteRT compatibility rewrite changed the network output "
            f"too much: max_abs_diff={max_abs_diff:.9g}"
        )

    return changes, max_abs_diff


def convert_to_litert(
    model: nn.Module,
    example_input: torch.Tensor,
    output_tflite: Path,
    output_pt2: Path,
) -> tuple[str, int]:
    try:
        import litert_torch
    except ModuleNotFoundError as exc:
        if exc.name == "litert_torch":
            raise RuntimeError(
                "litert_torch is not installed.\n"
            ) from exc

        raise RuntimeError(
            "litert_torch is installed, but one of its dependencies "
            f"could not be imported: {exc}"
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "litert_torch is installed, but importing it failed because "
            f"of a binary/dependency error: {exc}"
        ) from exc

    try:
        import importlib.metadata as metadata
        litert_version = metadata.version(
            "litert-torch"
        )
    except Exception:
        litert_version = "unknown"

    wrapped = TensorOutputWrapper(
        model
    ).cpu().eval()

    print(
        "Running torch.export sanity check...",
        flush=True,
    )

    exported = torch.export.export(
        wrapped,
        (example_input,),
    )

    if SAVE_PT2:
        torch.export.save(
            exported,
            str(output_pt2),
        )

        print(
            f"Saved PT2: {output_pt2}",
            flush=True,
        )

    print(
        f"LiteRT Torch version: {litert_version}",
        flush=True,
    )
    print(
        "Converting PyTorch -> LiteRT/TFLite...",
        flush=True,
    )

    edge_model = litert_torch.convert(
        wrapped,
        (example_input,),
    )

    edge_model.export(
        str(output_tflite)
    )

    if (
        not output_tflite.is_file()
        or output_tflite.stat().st_size == 0
    ):
        raise RuntimeError(
            "LiteRT did not create a valid .tflite file."
        )

    return (
        litert_version,
        output_tflite.stat().st_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one arbitrary nnUNetTrainer_*.py file "
            "to LiteRT/TFLite."
        )
    )

    parser.add_argument(
        "trainer",
        type=Path,
        help=(
            "Path to a single nnUNetTrainer_*.py file."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional trained checkpoint. Not required for "
            "latency-only conversion."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    trainer_path = args.trainer.expanduser().resolve()

    if not trainer_path.is_file():
        raise FileNotFoundError(
            f"Trainer file does not exist: {trainer_path}"
        )

    torch.manual_seed(
        TORCH_SEED
    )

    class_name = detect_trainer_class_name(
        trainer_path
    )

    trainer_stem = trainer_path.stem

    output_directory = (
        OUTPUT_ROOT / trainer_stem
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_tflite = (
        output_directory
        / f"{trainer_stem}_256x256.tflite"
    )
    output_pt2 = (
        output_directory
        / f"{trainer_stem}_256x256.pt2"
    )
    output_report = (
        output_directory
        / f"{trainer_stem}_256x256_conversion.txt"
    )

    print("=" * 80)
    print("GENERIC nnU-Net TRAINER -> LiteRT/TFLite")
    print("=" * 80)
    print(
        f"Trainer file : {trainer_path}"
    )
    print(
        f"Detected cls : {class_name}"
    )
    print(
        f"Input        : "
        f"{BATCH_SIZE} x {INPUT_CHANNELS} x "
        f"{HEIGHT} x {WIDTH}"
    )
    print(
        f"Output chans : {OUTPUT_CHANNELS}"
    )
    print(
        f"TFLite out   : {output_tflite}"
    )
    print()

    print(
        "Importing trainer...",
        flush=True,
    )

    trainer_module = import_trainer_script(
        trainer_path
    )

    trainer_class = get_trainer_class(
        trainer_module,
        class_name,
    )

    print(
        "Building network...",
        flush=True,
    )

    model = build_network(
        trainer_class
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters   : {parameter_count:,}",
        flush=True,
    )

    checkpoint_description = (
        load_checkpoint_if_requested(
            model,
            args.checkpoint,
        )
    )

    print(
        f"Checkpoint   : {checkpoint_description}",
        flush=True,
    )

    example_input = torch.randn(
        BATCH_SIZE,
        INPUT_CHANNELS,
        HEIGHT,
        WIDTH,
        dtype=INPUT_DTYPE,
    )

    print(
        "Running local forward...",
        flush=True,
    )

    with torch.inference_mode():
        local_output = primary_output(
            model(example_input)
        )

    print(
        f"Local output : {tuple(local_output.shape)}",
        flush=True,
    )

    print(
        "Checking generic LiteRT compatibility rewrites...",
        flush=True,
    )

    compatibility_changes, compatibility_max_abs_diff = (
        apply_generic_litert_compatibility(
            model,
            example_input,
        )
    )

    if compatibility_changes:
        print(
            "Applied exact fixed-shape compatibility rewrites:",
            flush=True,
        )
        for change in compatibility_changes:
            print(
                f"  - {change}",
                flush=True,
            )
        print(
            f"Output max abs diff after rewrite: "
            f"{compatibility_max_abs_diff:.9g}",
            flush=True,
        )
    else:
        print(
            "No generic compatibility rewrite was required.",
            flush=True,
        )

    litert_version, tflite_size = (
        convert_to_litert(
            model=model,
            example_input=example_input,
            output_tflite=output_tflite,
            output_pt2=output_pt2,
        )
    )

    tflite_size_mib = (
        tflite_size / (1024 ** 2)
    )

    report_lines = [
        "Generic nnU-Net -> LiteRT conversion report",
        "=" * 80,
        f"Trainer file    : {trainer_path}",
        f"Trainer class   : {class_name}",
        f"Parameters      : {parameter_count:,}",
        f"Checkpoint      : {checkpoint_description}",
        (
            "Input           : "
            f"{BATCH_SIZE} x {INPUT_CHANNELS} x "
            f"{HEIGHT} x {WIDTH}"
        ),
        (
            "Output          : "
            + " x ".join(
                str(value)
                for value in local_output.shape
            )
        ),
        f"Input dtype     : {INPUT_DTYPE}",
        f"LiteRT Torch    : {litert_version}",
        f"TFLite file     : {output_tflite}",
        f"TFLite size     : {tflite_size_mib:.3f} MiB",
        (
            f"PT2 file        : "
            f"{output_pt2 if SAVE_PT2 else 'disabled'}"
        ),
    ]

    if compatibility_changes:
        report_lines += [
            "",
            "Compatibility rewrites:",
            *[
                f"  - {change}"
                for change in compatibility_changes
            ],
        ]

    output_report.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("CONVERSION SUCCESSFUL")
    print("=" * 80)
    print(
        f"TFLite : {output_tflite}"
    )
    print(
        f"Size   : {tflite_size_mib:.3f} MiB"
    )
    print(
        f"Report : {output_report}"
    )

    if SAVE_PT2:
        print(
            f"PT2    : {output_pt2}"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except Exception as exc:
        print()
        print(
            "=" * 80,
            file=sys.stderr,
        )
        print(
            "CONVERSION FAILED",
            file=sys.stderr,
        )
        print(
            "=" * 80,
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print()
        traceback.print_exc()

        raise SystemExit(1)
