import math
from pathlib import Path

import numpy as np
import torch

from .angle_transforms import AngleBatchPreprocessor
from .network import AnglePredictor, LineAngleModel
from .polar_refinement import PolarRefinementModel


def load_model(checkpoint: str | Path, device="cpu"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    params = metadata["params"]
    config = params["model"]
    if config["kind"] != "polar_refinement":
        raise ValueError("This package expects the selected polar refinement model")
    coarse_options = dict(config["coarse_model"])
    coarse_options.pop("kind")
    coarse_options["pretrained"] = False
    coarse = LineAngleModel(**coarse_options)
    network = PolarRefinementModel(
        coarse,
        crop_size=tuple(config["crop_size"]),
        window_deg=config["window_deg"],
        fine_channels=tuple(config["fine_channels"]),
        refinement_hidden=config["refinement_hidden"],
        refinement_head=config.get("refinement_head", "mlp"),
        detach_crop_angle=config.get("detach_crop_angle", False),
        train_coarse=False,
        fine_angular_kernel=config["fine_angular_kernel"],
        fine_radial_kernel=config["fine_radial_kernel"],
        fine_radial_strides=(
            tuple(config["fine_radial_strides"])
            if config["fine_radial_strides"] is not None
            else None
        ),
        radial_pool_bins=config["radial_pool_bins"],
        fine_radial_antialias=config["fine_radial_antialias"],
    )
    input_size = tuple(metadata["input_size"])
    preprocess_options = dict(params["preprocessing"])
    preprocess_options["output_size"] = tuple(preprocess_options["output_size"])
    model = AnglePredictor(
        AngleBatchPreprocessor(input_size=input_size, **preprocess_options),
        network,
        inference_precision=params["training"]["precision"],
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(torch.device(device)).eval().requires_grad_(False)
    return model, input_size


def predict_rgb(model, pixels: np.ndarray) -> tuple[float, list[float]]:
    image = torch.from_numpy(np.asarray(pixels, dtype=np.uint8).copy()).permute(2, 0, 1)
    device = next(model.parameters()).device
    use_bf16 = model.inference_precision == "bf16" and device.type == "cuda"
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16),
    ):
        vector = model(image.unsqueeze(0).to(device)).float().cpu()[0]
    if not torch.isfinite(vector).all() or vector.norm() < 1e-6:
        raise ValueError("Model produced an undefined angle")
    angle = (0.5 * torch.atan2(vector[0], vector[1])).remainder(math.pi)
    return float(angle * (180 / math.pi)), vector.tolist()
