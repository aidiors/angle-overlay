from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tensorrt_rtx as trt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=Path("models/angle-bf16.onnx"))
    args = parser.parse_args()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(args.onnx.resolve())):
        raise RuntimeError(
            "\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors))
        )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT-RTX did not produce an engine")
    output = args.onnx.with_suffix(".rtx")
    output.write_bytes(bytes(serialized))
    metadata = json.loads(args.onnx.with_suffix(".json").read_text())
    metadata["runtime"] = "TensorRT-RTX"
    metadata["runtime_version"] = trt.__version__
    metadata["tf32"] = False
    metadata["hardware_target"] = "portable default RTX Ampere/Ada/Blackwell"
    metadata["files"][output.name] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("Built portable engine:", output)


if __name__ == "__main__":
    main()
