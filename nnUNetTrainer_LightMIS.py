from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch._dynamo import OptimizedModule
except Exception:
    OptimizedModule = type(None)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class PWConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1),
        ]
        if act:
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AKF(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.local = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.dilated = nn.Conv2d(
            channels,
            channels,
            3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False,
        )
        self.gate = nn.Conv2d(channels, channels, 1, bias=True)
        self.fuse = PWConv(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        dilated = self.dilated(x)
        alpha = torch.sigmoid(
            self.gate(F.adaptive_avg_pool2d(x, output_size=1))
        )
        return self.fuse(alpha * local + (1.0 - alpha) * dilated)


class ResAKF(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.akf = AKF(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.akf(x)


class PRFCore(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = 2 * channels

        c1 = max(1, hidden // 4)
        c2 = max(1, hidden // 4)
        c3 = hidden - c1 - c2
        self.splits = (c1, c2, c3)

        self.expand = PWConv(channels, hidden)

        specifications = (
            (c1, 3, 1, 1),
            (c2, 3, 2, 2),
            (c3, 5, 2, 1),
        )

        branches = []
        for width, kernel, padding, dilation in specifications:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        width,
                        width,
                        kernel,
                        padding=padding,
                        dilation=dilation,
                        groups=width,
                        bias=False,
                    ),
                    nn.BatchNorm2d(width, momentum=0.1),
                    nn.GELU(),
                )
            )
        self.dw1, self.dw2, self.dw3 = branches
        self.project = PWConv(hidden, channels, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)
        a, b, c = torch.split(y, self.splits, dim=1)

        a = self.dw1(a)
        b = self.dw2(b + a.mean(dim=1, keepdim=True))
        c = self.dw3(c + b.mean(dim=1, keepdim=True))

        return self.project(torch.cat([a, b, c], dim=1))


class PRF(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.core = PRFCore(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.core(x)


class AFCBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.core = AKF(channels)
        self.refiner = PRF(channels)
        self.post_refiner_akf = ResAKF(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.core(x)
        x = self.refiner(x)
        return self.post_refiner_akf(x)


class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.low_proj = PWConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.low_proj(F.avg_pool2d(x, kernel_size=2, stride=2))


class LightMISHead(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        fuse_channels: int,
        num_classes: int,
    ):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)

        for index, channel in enumerate(self.channels):
            setattr(
                self,
                f"main_spatial{index}",
                nn.Sequential(
                    nn.Conv2d(
                        channel,
                        channel,
                        3,
                        padding=1,
                        groups=channel,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channel, momentum=0.1),
                    nn.GELU(),
                ),
            )
            setattr(
                self,
                f"proj{index}",
                PWConv(channel, fuse_channels),
            )

        self.main_mix = PWConv(
            fuse_channels * len(self.channels), fuse_channels
        )
        self.main_refiner = AKF(fuse_channels)
        self.refiner = PRF(fuse_channels)
        self.final_head_akf = ResAKF(fuse_channels)
        self.classifier = nn.Sequential(
            PWConv(fuse_channels, fuse_channels),
            nn.Conv2d(fuse_channels, num_classes, 1),
        )

    @staticmethod
    def _align(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=ref.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        ref = features[1]
        main_parts = [
            self._align(
                getattr(self, f"proj{i}")(
                    getattr(self, f"main_spatial{i}")(feature)
                ),
                ref,
            )
            for i, feature in enumerate(features)
        ]
        main = self.main_mix(torch.cat(main_parts, dim=1))
        main = self.main_refiner(main)
        main = self.refiner(main)
        main = self.final_head_akf(main)
        logits = self.classifier(main)
        return F.interpolate(
            logits,
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )


class LightMISNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        deep_supervised: bool = False,
    ):
        super().__init__()
        self.channels = (10, 20, 32, 48, 80)
        self.deep_supervised = bool(deep_supervised)
        c0, c1, c2, c3, c4 = self.channels

        self.stem = Stem(input_channels, c0)
        self.b0 = AFCBlock(c0)
        self.d0 = DownSample(c0, c1)
        self.b1 = AFCBlock(c1)
        self.d1 = DownSample(c1, c2)
        self.b2 = AFCBlock(c2)
        self.d2 = DownSample(c2, c3)
        self.b3 = nn.Sequential(AFCBlock(c3))
        self.d3 = DownSample(c3, c4)
        self.b4 = nn.Sequential(AFCBlock(c4))

        self.head = LightMISHead(
            channels=self.channels,
            fuse_channels=32,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[2:]

        e0 = self.b0(self.stem(x))
        e1 = self.b1(self.d0(e0))
        e2 = self.b2(self.d1(e1))
        e3 = self.b3(self.d2(e2))
        e4 = self.b4(self.d3(e3))

        return self.head([e0, e1, e2, e3, e4], out_size)


class nnUNetTrainer_LightMIS(
    nnUNetTrainer
):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            device,
        )
        self.enable_deep_supervision = False
        self.initial_lr = 1e-2
        self.num_epochs = 200

    def set_deep_supervision_enabled(self, enabled: bool):
        model = self.network.module if self.is_ddp else self.network
        if isinstance(model, OptimizedModule):
            model = model._orig_mod
        model.deep_supervised = enabled

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    ):
        from dynamic_network_architectures.initialization.weight_init import (
            InitWeights_He,
        )

        model = LightMISNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervised=enable_deep_supervision,
        )
        model.apply(InitWeights_He(1e-2))
        for module in model.modules():
            if isinstance(module, (PRF, ResAKF)):
                nn.init.constant_(module.gamma, 0.05)
        return model
