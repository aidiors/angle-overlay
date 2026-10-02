from __future__ import annotations

import logging
import os
import re
from typing import Any

import numpy as np
from PIL import ImageGrab

from .monitors import Monitor, physical_dpi_context, primary_monitor


def crop_bounds(x: int, y: int, size: tuple[int, int], width: int, height: int):
    h, w = size
    left, top = x - w // 2, y - h // 2
    box = max(0, left), max(0, top), min(width, left + w), min(height, top + h)
    pads = ((box[1] - top, top + h - box[3]), (box[0] - left, left + w - box[2]), (0, 0))
    return box, pads


def primary_output() -> tuple[int, int]:
    import dxcam

    for line in dxcam.output_info().splitlines():
        match = re.search(r"Device\[(\d+)\] Output\[(\d+)\].*Primary:True", line)
        if match:
            return int(match[1]), int(match[2])
    raise RuntimeError("DXcam did not report a primary monitor")


def monitor_output(monitor: Monitor) -> tuple[int, int]:
    import dxcam

    # Match DXGI device names because DXcam omits headless adapters from its indices
    factory = getattr(dxcam, "__factory")
    for device, outputs in enumerate(factory.outputs):
        for index, output in enumerate(outputs):
            output.update_desc()
            if output.attached_to_desktop and output.devicename == monitor.device:
                return device, index
    raise RuntimeError(f"DXcam cannot map display {monitor.device}; using MSS")


def create_camera(backend: str, monitor: Monitor | None = None):
    import dxcam

    os.environ["DXCAM_WINRT_CURSOR_CAPTURE"] = "0"
    if backend not in {"dxgi", "winrt"}:
        raise ValueError("Camera backend must be dxgi or winrt")
    device, output = monitor_output(monitor) if monitor is not None else primary_output()
    return dxcam.create(
        device_idx=device,
        output_idx=output,
        output_color="BGRA",
        backend=backend,
        processor_backend="numpy",
        max_buffer_len=4,
    )


class ScreenCapture:
    def __init__(
        self,
        backend: str,
        width: int,
        height: int,
        fps: int = 120,
        *,
        monitor: Monitor | None = None,
    ):
        physical_dpi_context()
        self.monitor = monitor or primary_monitor()
        self.backend = backend
        self.width, self.height = width, height
        self.camera: Any = None
        self.mss: Any = None
        self.full_stream = backend.endswith("-stream")
        self.fallbacks = 0
        camera_backend = backend.removesuffix("-stream")
        if camera_backend in {"dxgi", "winrt"}:
            try:
                self.camera = create_camera(camera_backend, self.monitor)
                if (self.camera.width, self.camera.height) != (width, height):
                    raise RuntimeError("Capture dimensions do not match the selected display")
                if self.full_stream:
                    self.camera.start(target_fps=fps, video_mode=False)
            except Exception:
                logging.exception(
                    "Camera unavailable for %s; falling back to MSS", self.monitor.device
                )
                if self.camera is not None:
                    self.camera.release()
                    self.camera = None
                self.backend = "mss"
                self.full_stream = False
            if not self.full_stream:
                import mss

                self.mss = mss.MSS()
        elif backend == "mss":
            import mss

            self.mss = mss.MSS()
        elif backend != "pillow":
            raise ValueError(f"Unknown capture backend: {backend}")

    def crop(self, x: int, y: int, size: tuple[int, int]) -> np.ndarray | None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("Cursor is outside the selected display")
        box, pads = crop_bounds(x, y, size, self.width, self.height)
        desktop_box = self.monitor.desktop_box(box)
        if self.camera is not None:
            if self.full_stream:
                # Copy the frame before the producer reuses its ring buffer
                frame = self.camera.grab(copy=True)
                if frame is None:
                    return None
                pixels = frame[box[1] : box[3], box[0] : box[2], 2::-1]
            else:
                frame = self.camera.grab(region=box, copy=True, new_frame_only=False)
                if frame is None:
                    assert self.mss is not None
                    frame = np.asarray(self.mss.grab(desktop_box))
                    self.fallbacks += 1
                pixels = frame[:, :, 2::-1]
        elif self.mss is not None:
            pixels = np.asarray(self.mss.grab(desktop_box))[:, :, 2::-1]
        else:
            pixels = np.asarray(
                ImageGrab.grab(
                    bbox=desktop_box, all_screens=True, include_layered_windows=False
                ).convert("RGB")
            )
        if pixels.shape[:2] != (box[3] - box[1], box[2] - box[0]):
            raise RuntimeError("Capture was scaled instead of returning physical pixels")
        return np.pad(pixels, pads, mode="edge").copy()

    def close(self) -> None:
        if self.camera is not None:
            self.camera.release()
        if self.mss is not None:
            self.mss.close()
