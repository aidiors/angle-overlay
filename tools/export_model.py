from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch
from torch import nn

from angle_overlay.torch_model.loader import load_model


class ExportModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        polar = self.model.preprocessor.polar(rgb.float() / 255.0)
        polar = (polar - self.model.preprocessor.mean) / self.model.preprocessor.std
        return self.model.network(polar)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/source.pt"))
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--target", choices=("default", "directml"), default="default")
    args = parser.parse_args()
    if args.target == "directml" and args.precision != "fp32":
        parser.error("The DirectML export preserves FP32 precision")
    # Opset17 keeps GELU and GridSample on the GPU with the pinned DirectML runtime
    opset = 17 if args.target == "directml" else 20
    device = "cuda" if args.precision == "bf16" else "cpu"
    torch.set_num_threads(4)
    model, size = load_model(args.checkpoint, device)
    export = ExportModel(model).eval().requires_grad_(False)
    sample = torch.zeros((1, 3, *size), dtype=torch.float32, device=device)
    name = "angle-directml" if args.target == "directml" else f"angle-{args.precision}"
    output = Path("models") / f"{name}.onnx"
    with (
        torch.inference_mode(),
        torch.autocast(device, dtype=torch.bfloat16, enabled=args.precision == "bf16"),
    ):
        torch.onnx.export(
            export,
            (sample,),
            output,
            input_names=["rgb"],
            output_names=["vector"],
            opset_version=opset,
            dynamo=False,
            external_data=False,
        )
    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    graph.doc_string = ""
    for node in graph.graph.node:
        node.doc_string = ""
    onnx.save(graph, output)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = payload["metadata"]["params"]["model"]
    manifest = {
        "format_version": 1,
        "architecture": "ConvNeXt Tiny + frozen fine CNN128",
        "input_shape": [1, 3, *size],
        "input_dtype": "float32",
        "input_range": "RGB0..255",
        "output": "[sin(2theta),cos(2theta)]",
        "crop_size": config["crop_size"],
        "window_deg": config["window_deg"],
        "fine_channels": config["fine_channels"],
        "fine_angular_kernel": config["fine_angular_kernel"],
        "fine_radial_antialias": config["fine_radial_antialias"],
        "source_run_id": payload["metadata"]["run_id"],
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "precision": args.precision,
        "opset": opset,
        "target": args.target,
        "files": {output.name: hashlib.sha256(output.read_bytes()).hexdigest()},
        "release_validation": "pending compiled-runtime validation",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Exported", output, output.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
