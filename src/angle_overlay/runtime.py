from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort

from .devices import hardware_adapters


def decode_vector(vector: np.ndarray) -> tuple[float, list[float]]:
    vector = np.asarray(vector, dtype=np.float32).reshape(2)
    if not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-6:
        raise ValueError("The model produced an undefined angle")
    angle = math.degrees(0.5 * math.atan2(float(vector[0]), float(vector[1]))) % 180
    return angle, vector.tolist()


def rgb_input(pixels: np.ndarray) -> np.ndarray:
    if pixels.shape != (256, 256, 3) or pixels.dtype != np.uint8:
        raise ValueError("Expected owned RGB uint8 pixels of shape256×256×3")
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)


class Predictor(Protocol):
    device_label: str

    def predict(self, pixels: np.ndarray) -> tuple[float, list[float]]: ...
    def close(self) -> None: ...


class CPUPredictor:
    device_label = "CPU (ONNX Runtime / FP32)"

    def __init__(
        self, model: Path, *, directml_device: int | None = None, profile_prefix: Path | None = None
    ):
        manifest = json.loads(model.with_suffix(".json").read_text())
        if hashlib.sha256(model.read_bytes()).hexdigest() != manifest["files"][model.name]:
            raise RuntimeError("ONNX model checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
        if directml_device is not None:
            if directml_device < 0:
                raise ValueError("DirectML adapter index must be nonnegative")
            if "DmlExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("This build does not include the DirectML GPU runtime")
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            providers = ["DmlExecutionProvider"]
            provider_options = [{"device_id": str(directml_device)}]
            self.device_label = f"DirectML GPU {directml_device} (FP32)"
        if profile_prefix is not None:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_prefix)
        self.session = ort.InferenceSession(
            str(model), options, providers=providers, provider_options=provider_options
        )
        self.session.disable_fallback()
        if (
            directml_device is not None
            and "DmlExecutionProvider" not in self.session.get_providers()
        ):
            raise RuntimeError("DirectML initialization fell back to CPU")
        if self.session.get_inputs()[0].shape != [1, 3, 256, 256]:
            raise ValueError("Unexpected model input shape")
        # DirectML forbids concurrent Run calls on one session
        self.lock = threading.Lock()

    def predict(self, pixels: np.ndarray) -> tuple[float, list[float]]:
        prepared = rgb_input(pixels)
        with self.lock:
            if self.session is None:
                raise RuntimeError("The predictor is closed")
            vector = self.session.run(["vector"], {"rgb": prepared})[0]
        if not isinstance(vector, np.ndarray):
            raise RuntimeError("Unexpected inference output")
        return decode_vector(vector)

    def close(self) -> None:
        with self.lock:
            self.session = None


class DirectMLPredictor(CPUPredictor):
    def __init__(self, model: Path, device_id: int = 0, *, profile_prefix: Path | None = None):
        super().__init__(model, directml_device=device_id, profile_prefix=profile_prefix)


def _create_predictor(models: Path, backend: str, directml_device: int) -> Predictor:
    if backend == "cpu":
        return CPUPredictor(models / "angle-fp32.onnx")
    if backend == "directml":
        return DirectMLPredictor(models / "angle-directml.onnx", directml_device)
    if backend == "rtx":
        from .rtx_runtime import RTXPredictor

        return RTXPredictor(models / "angle-bf16.rtx")
    raise ValueError(f"Unknown runtime: {backend}")


def _automatic_candidates() -> tuple[str, ...]:
    try:
        adapters = hardware_adapters()
    except Exception:
        logging.exception("Adapter discovery failed; probing available runtimes")
        return ("rtx", "directml", "cpu")
    if not any(adapter.vendor_id == 0x10DE for adapter in adapters):
        return ("directml", "cpu")
    return ("rtx", "directml", "cpu")


def load_predictor(models: Path, backend: str, directml_device: int = 0) -> Predictor:
    if backend not in {"auto", "cpu", "rtx", "directml"}:
        raise ValueError(f"Unknown runtime: {backend}")
    candidates = _automatic_candidates() if backend == "auto" else (backend,)
    for candidate in candidates:
        predictor = None
        try:
            predictor = _create_predictor(models, candidate, directml_device)
            predictor.predict(np.zeros((256, 256, 3), dtype=np.uint8))
        except Exception:
            if predictor is not None:
                predictor.close()
            if backend != "auto" or candidate == "cpu":
                raise
            logging.exception("%s initialization/warmup failed; trying next runtime", candidate)
        else:
            if backend == "auto" and candidate == "cpu":
                predictor.device_label += " — GPU startup failed; see app.log"
            return predictor
    raise RuntimeError("No inference runtime could start")
