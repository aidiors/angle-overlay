from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .paths import data_directory
from .runtime import decode_vector, rgb_input


def cuda_check(result):
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA operation failed: {error}")
    return values[0] if len(values) == 1 else values


class RTXPredictor:
    device_label = "NVIDIA RTX (TensorRT-RTX / BF16)"

    def __init__(self, engine_file: Path):
        # Load NVIDIA DLLs only when RTX is selected so AMD and CPU modes can start
        if getattr(sys, "frozen", False):
            bundle_path = getattr(sys, "_MEIPASS", None)
            if bundle_path is None:
                raise RuntimeError("The frozen RTX bundle is missing its library directory")
            bundle = Path(bundle_path)
            self._cudart_library = ctypes.WinDLL(str(bundle / "cudart64_13.dll"))
        trt: Any = importlib.import_module("tensorrt_rtx")
        cuda: Any = importlib.import_module("cuda.bindings.runtime")

        self.cuda = cuda
        self.device_allocations = []
        self.host_allocations = []
        self.stream = None
        self.closed = False
        manifest = json.loads(engine_file.with_suffix(".manifest.json").read_text())
        if manifest["runtime_version"] != trt.__version__:
            raise RuntimeError("Model and TensorRT-RTX versions differ; use a matching release")
        blob = engine_file.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != manifest["files"][engine_file.name]:
            raise RuntimeError("RTX model checksum mismatch")
        cuda_check(cuda.cudaSetDevice(0))
        cuda_check(cuda.cudaFree(0))
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError("This RTX model could not be loaded on the current GPU/driver")
        for name, shape in [("rgb", (1, 3, 256, 256)), ("vector", (1, 2))]:
            if tuple(self.engine.get_tensor_shape(name)) != shape:
                raise RuntimeError(f"Unexpected engine tensor: {name}")
            if self.engine.get_tensor_dtype(name) != trt.float32:
                raise RuntimeError(f"Expected FP32 engine IO: {name}")
        self.runtime_config = self.engine.create_runtime_config()
        self.cache = self.runtime_config.create_runtime_cache()
        cache_dir = data_directory() / "cache"
        cache_dir.mkdir(exist_ok=True)
        self.cache_path = cache_dir / f"{digest}-{trt.__version__}.cache"
        if self.cache_path.exists():
            self.cache.deserialize(self.cache_path.read_bytes())
        self.runtime_config.set_runtime_cache(self.cache)
        self.context = self.engine.create_execution_context(self.runtime_config)
        if self.context is None:
            raise RuntimeError("RTX initialization failed; check the NVIDIA driver")
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_bytes(bytes(self.cache.serialize()))
        temporary.replace(self.cache_path)
        try:
            self.stream = cuda_check(cuda.cudaStreamCreate())
            self.host_input = self.allocate(3 * 256 * 256 * 4, host=True)
            self.host_output = self.allocate(8, host=True)
            self.device_input = self.allocate(3 * 256 * 256 * 4, host=False)
            self.device_output = self.allocate(8, host=False)
        except Exception:
            self.close()
            raise
        self.input_array = np.ctypeslib.as_array(
            (ctypes.c_float * (3 * 256 * 256)).from_address(self.host_input)
        ).reshape(1, 3, 256, 256)
        self.output_array = np.ctypeslib.as_array(
            (ctypes.c_float * 2).from_address(self.host_output)
        )
        assert self.context.set_tensor_address("rgb", self.device_input)
        assert self.context.set_tensor_address("vector", self.device_output)

    def allocate(self, size: int, *, host: bool):
        function = self.cuda.cudaMallocHost if host else self.cuda.cudaMalloc
        address = cuda_check(function(size))
        (self.host_allocations if host else self.device_allocations).append(address)
        return address

    def predict(self, pixels: np.ndarray) -> tuple[float, list[float]]:
        if self.closed or self.context is None:
            raise RuntimeError("Predictor is closed")
        cuda = self.cuda
        cuda_check(cuda.cudaSetDevice(0))
        np.copyto(self.input_array, rgb_input(pixels))
        cuda_check(
            cuda.cudaMemcpyAsync(
                self.device_input,
                self.host_input,
                self.input_array.nbytes,
                cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )
        )
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("RTX inference failed")
        cuda_check(
            cuda.cudaMemcpyAsync(
                self.host_output,
                self.device_output,
                8,
                cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
        )
        cuda_check(cuda.cudaStreamSynchronize(self.stream))
        return decode_vector(self.output_array)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.stream is not None:
            cuda_check(self.cuda.cudaSetDevice(0))
            cuda_check(self.cuda.cudaStreamSynchronize(self.stream))
            for address in self.device_allocations:
                self.cuda.cudaFree(address)
            for address in self.host_allocations:
                self.cuda.cudaFreeHost(address)
            self.cuda.cudaStreamDestroy(self.stream)
        self.context = None
        self.engine = None
        self.runtime = None
