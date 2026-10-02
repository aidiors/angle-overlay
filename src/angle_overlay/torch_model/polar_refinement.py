from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


def crop_signed_polar(
    image: torch.Tensor,
    angle: torch.Tensor,
    *,
    crop_size: tuple[int, int],
    window_deg: float,
) -> torch.Tensor:
    height, width = crop_size
    offsets = torch.linspace(
        -math.radians(window_deg),
        math.radians(window_deg),
        width,
        device=image.device,
        dtype=torch.float32,
    )
    columns = image.shape[-1] + (angle[:, None].float() + offsets) * (image.shape[-1] / math.pi)
    extended = torch.cat((image.flip(-2), image, image.flip(-2)), dim=-1)
    x = (2 * columns / (extended.shape[-1] - 1) - 1)[:, None].expand(-1, height, -1)
    radius = torch.linspace(-1, 1, height, device=image.device, dtype=torch.float32)
    y = radius[None, :, None].expand(len(image), -1, width)
    grid = torch.stack((x, y), dim=-1)
    # FP32 sampling preserves subpixel precision under BF16 autocast
    with torch.amp.autocast(device_type=image.device.type, enabled=False):
        return F.grid_sample(
            extended.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
        )


def rotate_double_angle(vector: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    sine, cosine = torch.sin(2 * offset), torch.cos(2 * offset)
    return torch.stack(
        (vector[:, 0] * cosine + vector[:, 1] * sine, vector[:, 1] * cosine - vector[:, 0] * sine),
        dim=-1,
    )


def angular_heatmap_offset(scores: torch.Tensor, positive_offsets: torch.Tensor) -> torch.Tensor:
    probabilities = scores.float().softmax(-1)
    half = positive_offsets.numel()
    # Pair symmetric probabilities so uniform scores produce exactly zero offset
    imbalance = probabilities[..., half + 1 :] - probabilities[..., :half].flip(-1)
    return (imbalance * positive_offsets).sum(-1)


def radial_feature_profile(features: torch.Tensor, bins: int = 1) -> torch.Tensor:
    values = features.float()
    if bins == 1:
        # Keep reduction order unchanged for checkpoint reproducibility
        return torch.cat((values.mean(-2), values.amax(-2)), dim=1)
    size = (bins, values.shape[-1])
    means = F.adaptive_avg_pool2d(values, size)
    maxima = F.adaptive_max_pool2d(values, size)
    return torch.cat((means, maxima), dim=1).flatten(1, 2)


class RadialAntialias(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        padded = F.pad(features, (0, 0, 1, 1), mode="replicate")
        return (padded[..., :-2, :] + 2 * padded[..., 1:-1, :] + padded[..., 2:, :]) * 0.25


class PolarRefinementModel(nn.Module):
    def __init__(
        self,
        coarse: nn.Module,
        crop_size: tuple[int, int] = (192, 33),
        window_deg: float = 3.0,
        fine_channels: tuple[int, ...] = (16, 32, 32),
        refinement_hidden: int = 128,
        train_coarse: bool = False,
        refinement_head: Literal["mlp", "local_mlp", "heatmap"] = "mlp",
        detach_crop_angle: bool = False,
        fine_angular_kernel: int = 5,
        radial_pool_bins: int = 1,
        fine_radial_antialias: bool = False,
        fine_radial_kernel: int = 5,
        fine_radial_strides: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if fine_angular_kernel <= 0 or fine_angular_kernel % 2 != 1:
            raise ValueError("fine_angular_kernel must be positive and odd")
        if fine_radial_kernel <= 0 or fine_radial_kernel % 2 != 1:
            raise ValueError("fine_radial_kernel must be positive and odd")
        strides = (2,) * len(fine_channels) if fine_radial_strides is None else fine_radial_strides
        if len(strides) != len(fine_channels) or any(stride not in (1, 2) for stride in strides):
            raise ValueError("fine_radial_strides must match fine_channels with values1 or2")
        radial_rows = math.ceil(crop_size[0] / math.prod(strides))
        if radial_pool_bins <= 0 or radial_pool_bins > radial_rows:
            raise ValueError("radial_pool_bins must fit the fine feature radial rows")
        self.radial_pool_bins = radial_pool_bins
        self.train_coarse = train_coarse
        self.coarse = coarse.requires_grad_(train_coarse)
        if not train_coarse:
            self.coarse.eval()
        self.crop_size = crop_size
        self.window_deg = window_deg
        self.refinement_head = refinement_head
        self.detach_crop_angle = detach_crop_angle
        blocks: list[nn.Module] = []
        channels = 5
        for out_channels, radial_stride in zip(fine_channels, strides, strict=True):
            if fine_radial_antialias:
                blocks.append(RadialAntialias())
            blocks.extend(
                [
                    nn.Conv2d(
                        channels,
                        out_channels,
                        (fine_radial_kernel, fine_angular_kernel),
                        stride=(radial_stride, 1),
                        padding=(fine_radial_kernel // 2, fine_angular_kernel // 2),
                    ),
                    nn.GroupNorm(math.gcd(4, out_channels), out_channels),
                    nn.SiLU(),
                ]
            )
            channels = out_channels
        self.fine = nn.Sequential(*blocks)
        if refinement_head in ("mlp", "local_mlp"):
            output_layer = nn.Linear(refinement_hidden, 1)
            conditioning_channels = 2 if refinement_head == "mlp" else 0
            self.correction = nn.Sequential(
                nn.Linear(
                    2 * channels * radial_pool_bins * crop_size[1] + conditioning_channels,
                    refinement_hidden,
                ),
                nn.GELU(),
                output_layer,
            )
        elif refinement_head == "heatmap":
            output_layer = nn.Conv1d(refinement_hidden, 1, 1)
            self.correction = nn.Sequential(
                nn.Conv1d(2 * channels * radial_pool_bins, refinement_hidden, 5, padding=2),
                nn.GELU(),
                output_layer,
            )
            offsets = torch.linspace(0, math.radians(window_deg), crop_size[1] // 2 + 1)[1:]
            self.register_buffer("positive_offsets", offsets)
        else:
            raise ValueError(f"Unknown refinement head: {refinement_head}")
        nn.init.zeros_(output_layer.weight)
        assert output_layer.bias is not None
        nn.init.zeros_(output_layer.bias)
        radius = torch.linspace(-1, 1, crop_size[0]).view(1, 1, -1, 1)
        self.register_buffer("radius", radius)

    def train(self, mode: bool = True) -> PolarRefinementModel:
        super().train(mode)
        if not self.train_coarse:
            self.coarse.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_coarse):
            coarse = self.coarse(image).float()
            angle = (0.5 * torch.atan2(coarse[:, 0], coarse[:, 1])).remainder(math.pi)
        crop_angle = angle.detach() if self.detach_crop_angle else angle
        crop = crop_signed_polar(
            image, crop_angle, crop_size=self.crop_size, window_deg=self.window_deg
        )
        radius_buffer = self.radius
        assert isinstance(radius_buffer, torch.Tensor)
        radius = radius_buffer.expand(len(image), -1, -1, self.crop_size[1])
        features = self.fine(torch.cat((crop, radius, radius.abs()), dim=1))
        with torch.amp.autocast(device_type=image.device.type, enabled=False):
            profile = radial_feature_profile(features, self.radial_pool_bins)
            if self.refinement_head == "heatmap":
                scores = self.correction(profile).squeeze(1)
                positive_offsets = self.positive_offsets
                assert isinstance(positive_offsets, torch.Tensor)
                offset = angular_heatmap_offset(scores, positive_offsets)
            else:
                hidden = profile.flatten(1)
                if self.refinement_head == "mlp":
                    hidden = torch.cat((hidden, coarse), dim=1)
                offset = torch.tanh(self.correction(hidden).squeeze(-1)) * math.radians(
                    self.window_deg
                )
            return rotate_double_angle(coarse, offset)
