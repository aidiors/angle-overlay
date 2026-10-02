from __future__ import annotations

import math
from typing import Any, Literal

import timm
import torch
from torch import nn
from torch.nn import functional as F

from .angle_transforms import AngleBatchPreprocessor
from .polar_refinement import PolarRefinementModel


class PolarAngleRegressionHead(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden: int = 256,
        dropout: float = 0.1,
        residual_scale: float = 0.25,
        normalize_output: bool = True,
        angular_dilations: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        if len(angular_dilations) != 3 or any(d <= 0 for d in angular_dilations):
            raise ValueError("angular dilations require three positive values")
        first, second, third = angular_dilations
        self.residual_scale = residual_scale
        self.normalize_output = normalize_output
        self.radius_query = nn.Parameter(torch.zeros(in_ch))
        self.angle_conv = nn.Sequential(
            nn.Conv1d(
                3 * in_ch + 2,
                hidden,
                5,
                padding=2 * first,
                dilation=first,
                padding_mode="circular",
            ),
            nn.GELU(),
            nn.Conv1d(
                hidden,
                hidden,
                5,
                padding=2 * second,
                dilation=second,
                padding_mode="circular",
            ),
            nn.GELU(),
            nn.Conv1d(
                hidden,
                hidden,
                3,
                padding=third,
                dilation=third,
                padding_mode="circular",
            ),
            nn.GELU(),
        )
        self.score = nn.Conv1d(hidden, 1, 1)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden + 2),
            nn.Linear(hidden + 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    @staticmethod
    def _angle_pos(
        batch_size: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        theta = torch.arange(width, device=device, dtype=dtype) * (math.pi / width)
        position = torch.stack((torch.sin(2 * theta), torch.cos(2 * theta)), dim=0)
        return position.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, _, width = features.shape
        mean_radius = features.mean(dim=2)
        max_radius = features.amax(dim=2)
        radius_scores = torch.einsum("bchw,c->bhw", features, self.radius_query)
        radius_attention = torch.softmax(radius_scores / math.sqrt(channels), dim=1).unsqueeze(1)
        attended_radius = (features * radius_attention).sum(dim=2)
        position = self._angle_pos(batch, width, features.device, features.dtype)
        hidden = self.angle_conv(
            torch.cat((mean_radius, max_radius, attended_radius, position), dim=1)
        )
        angle_attention = torch.softmax(self.score(hidden).squeeze(1).float(), dim=-1)
        angle_attention = angle_attention.to(hidden.dtype)
        pooled_features = (hidden * angle_attention.unsqueeze(1)).sum(dim=-1)
        pooled_position = (position * angle_attention.unsqueeze(1)).sum(dim=-1)
        delta = self.delta_head(torch.cat((pooled_features, pooled_position), dim=-1).float())
        output = pooled_position.float() + self.residual_scale * delta
        return F.normalize(output, p=2, dim=1, eps=1e-6) if self.normalize_output else output


class LineAngleModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        out_index: int = 3,
        head_hidden: int = 256,
        pretrained: bool = True,
        dropout: float = 0.1,
        residual_scale: float = 0.25,
        normalize_output: bool = True,
        final_stage_blocks: int | None = None,
        penultimate_stage_blocks: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone: Any = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(out_index,),
            in_chans=3,
        )
        if final_stage_blocks is not None:
            if (
                backbone_name != "convnext_tiny"
                or out_index != 3
                or not 1 <= final_stage_blocks <= 3
            ):
                raise ValueError(
                    "final_stage_blocks requires convnext_tiny stage 3 and 1..3 blocks"
                )
            stage = self.backbone.get_submodule("stages_3")
            stage.blocks = nn.Sequential(*list(stage.blocks.children())[:final_stage_blocks])
        if penultimate_stage_blocks is not None:
            if (
                backbone_name != "convnext_tiny"
                or out_index not in {2, 3}
                or not 1 <= penultimate_stage_blocks <= 9
            ):
                raise ValueError(
                    "penultimate_stage_blocks requires convnext_tiny stage 2 and 1..9 blocks"
                )
            stage = self.backbone.get_submodule("stages_2")
            stage.blocks = nn.Sequential(*list(stage.blocks.children())[:penultimate_stage_blocks])
        channels = self.backbone.feature_info.channels()[-1]
        self.head = PolarAngleRegressionHead(
            channels,
            hidden=head_hidden,
            dropout=dropout,
            residual_scale=residual_scale,
            normalize_output=normalize_output,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)[-1].float()
        if image.device.type in {"cuda", "cpu"}:
            with torch.amp.autocast(device_type=image.device.type, enabled=False):
                return self.head(features)
        return self.head(features)


class AnglePredictor(nn.Module):
    def __init__(
        self,
        preprocessor: AngleBatchPreprocessor,
        network: LineAngleModel | PolarRefinementModel,
        inference_precision: Literal["bf16", "fp32"] = "fp32",
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.network = network
        self.inference_precision = inference_precision

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        polar_images, _ = self.preprocessor(images, augment=False)
        return self.network(polar_images)

    def forward_augmented(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        polar_images, adjusted_targets = self.preprocessor(images, targets, augment=True)
        assert adjusted_targets is not None
        return self.network(polar_images), adjusted_targets
