from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def build_polar_grid(
    input_size: tuple[int, int] = (256, 256),
    output_size: tuple[int, int] = (384, 360),
) -> torch.Tensor:
    height, width = input_size
    height_out, width_out = output_size
    if min(height, width) <= 1 or min(height_out, width_out) <= 1:
        raise ValueError("Input and output dimensions must all exceed 1")

    cx, cy = (width - 1) / 2, (height - 1) / 2
    max_radius = min(cx, cy)
    theta = torch.arange(width_out, dtype=torch.float64) * (math.pi / width_out)
    radius = torch.linspace(-max_radius, max_radius, height_out, dtype=torch.float64)
    radius_grid, theta_grid = torch.meshgrid(radius, theta, indexing="ij")
    x = cx + radius_grid * torch.cos(theta_grid)
    y = cy + radius_grid * torch.sin(theta_grid)
    grid_x = 2 * x / (width - 1) - 1
    grid_y = 2 * y / (height - 1) - 1
    return torch.stack((grid_x, grid_y), dim=-1).clamp(-1, 1).float().unsqueeze(0)


class SignedPolarTransform(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int] = (256, 256),
        output_size: tuple[int, int] = (384, 360),
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.register_buffer("grid", build_polar_grid(input_size, output_size))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        single = image.ndim == 3
        if single:
            image = image.unsqueeze(0)
        if image.ndim != 4 or tuple(image.shape[-2:]) != self.input_size:
            raise ValueError(f"Expected [B,C,{self.input_size[0]},{self.input_size[1]}] image")
        if not image.is_floating_point():
            raise TypeError("SignedPolarTransform requires floating-point images")

        grid = self.grid
        assert isinstance(grid, torch.Tensor)
        grid = grid.to(device=image.device, dtype=image.dtype)
        grid = grid.expand(image.shape[0], -1, -1, -1)
        result = F.grid_sample(
            image, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return result.squeeze(0) if single else result


def pad_signed_polar_angle(image: torch.Tensor, pad: int) -> torch.Tensor:
    if image.ndim < 2 or not 0 <= pad < image.shape[-1]:
        raise ValueError("pad must be non-negative and smaller than angular width")
    if pad == 0:
        return image
    return torch.cat((image[..., :, -pad:].flip(-2), image, image[..., :, :pad].flip(-2)), dim=-1)


class AngleBatchPreprocessor(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int] = (256, 256),
        output_size: tuple[int, int] = (384, 360),
        *,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        jitter_probability: float = 0.7,
        gaussian_probability: float = 0.3,
        speckle_probability: float = 0.15,
    ) -> None:
        super().__init__()
        for name, probability in (
            ("horizontal_flip_probability", horizontal_flip_probability),
            ("vertical_flip_probability", vertical_flip_probability),
            ("jitter_probability", jitter_probability),
            ("gaussian_probability", gaussian_probability),
            ("speckle_probability", speckle_probability),
        ):
            if not 0 <= probability <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(x) for x in (*mean, *std)):
            raise ValueError("mean and std must contain three finite RGB values")
        if any(x <= 0 for x in std):
            raise ValueError("std values must be positive")

        self.polar = SignedPolarTransform(input_size, output_size)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.jitter_probability = jitter_probability
        self.gaussian_probability = gaussian_probability
        self.speckle_probability = speckle_probability

    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        augment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB images [B,3,H,W]")
        if images.dtype != torch.uint8:
            raise TypeError("Expected uint8 RGB images in the 0..255 range")
        if targets is not None and (targets.ndim != 2 or targets.shape != (len(images), 2)):
            raise ValueError("Expected targets [B,2] in [sin(2θ), cos(2θ)] order")
        if augment and targets is None:
            raise ValueError("Augmentation requires targets so flips can update angles")

        image = images.float().div_(255)
        label = targets.to(images.device) if targets is not None else None
        if augment:
            assert label is not None
            image, label = self._augment(image, label)
        image = self.polar(image)
        image = (image - self.mean) / self.std
        return image, label

    def _augment(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = image.shape[0]
        device = image.device
        horizontal = torch.rand(batch, device=device) < self.horizontal_flip_probability
        vertical = torch.rand(batch, device=device) < self.vertical_flip_probability
        image = torch.where(horizontal[:, None, None, None], image.flip(-1), image)
        image = torch.where(vertical[:, None, None, None], image.flip(-2), image)
        sign = torch.where(horizontal ^ vertical, -1.0, 1.0)
        label = torch.stack((label[:, 0] * sign, label[:, 1]), dim=1)

        jitter = (torch.rand(batch, device=device) < self.jitter_probability)[:, None, None, None]
        brightness = 0.9 + 0.2 * torch.rand(batch, 1, 1, 1, device=device)
        contrast = 0.9 + 0.2 * torch.rand(batch, 1, 1, 1, device=device)
        saturation = 0.9 + 0.2 * torch.rand(batch, 1, 1, 1, device=device)
        adjusted = image * brightness
        adjusted = (adjusted - adjusted.mean(dim=(1, 2, 3), keepdim=True)) * contrast + (
            adjusted.mean(dim=(1, 2, 3), keepdim=True)
        )
        weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        gray = (adjusted * weights).sum(dim=1, keepdim=True)
        adjusted = (adjusted - gray) * saturation + gray
        image = torch.where(jitter, adjusted, image)

        gaussian = torch.rand(batch, 1, 1, 1, device=device) < self.gaussian_probability
        sigma = 0.004 + 0.011 * torch.rand(batch, 1, 1, 1, device=device)
        image = image + gaussian * sigma * torch.randn_like(image)
        speckle = torch.rand(batch, 1, 1, 1, device=device) < self.speckle_probability
        strength = 0.005 + 0.015 * torch.rand(batch, 1, 1, 1, device=device)
        image = image + speckle * strength * image * torch.randn_like(image)
        return image.clamp_(0, 1), label
