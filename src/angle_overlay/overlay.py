from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import math
import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from contextlib import suppress
from ctypes import wintypes as wt
from pathlib import Path

import numpy as np
from PIL import ImageGrab

from angle_overlay.capture import ScreenCapture, crop_bounds
from angle_overlay.monitors import Monitor, physical_dpi_context, primary_monitor
from angle_overlay.paths import app_directory
from angle_overlay.runtime import load_predictor
from angle_overlay.settings import TRANSPARENT, line_color

HERE = app_directory()
STYLE_MASK = 0x00080000 | 0x00000020 | 0x08000000 | 0x00000080


def line_endpoints(x, y, angle_deg, width, height):
    dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    low, high = -math.inf, math.inf
    for point, direction, maximum in ((x, dx, width - 1), (y, dy, height - 1)):
        if abs(direction) < 1e-12:
            if not 0 <= point <= maximum:
                return None
        else:
            a, b = -point / direction, (maximum - point) / direction
            low, high = max(low, min(a, b)), min(high, max(a, b))
    if low > high:
        return None
    return (x + low * dx, y + low * dy, x + high * dx, y + high * dy)


def capture_crop(x, y, size, width, height, monitor: Monitor | None = None):
    box, pads = crop_bounds(x, y, size, width, height)
    if monitor is not None:
        box = monitor.desktop_box(box)
    image = ImageGrab.grab(bbox=box, all_screens=True, include_layered_windows=False).convert("RGB")
    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.shape[:2] != (box[3] - box[1], box[2] - box[0]):
        raise RuntimeError(
            "Capture dimensions differ from physical pixels; check DPI configuration"
        )
    return np.pad(pixels, pads, mode="edge").copy()


class Windows:
    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("This overlay requires Windows")
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.user.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        self.user.SetProcessDpiAwarenessContext.restype = wt.BOOL
        if not self.user.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            self.user.SetProcessDPIAware()
        physical_dpi_context()
        self.user.GetPhysicalCursorPos.argtypes = [ctypes.POINTER(wt.POINT)]
        self.user.GetPhysicalCursorPos.restype = wt.BOOL
        self.user.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user.GetAsyncKeyState.restype = ctypes.c_short
        self.user.GetAncestor.argtypes = [wt.HWND, wt.UINT]
        self.user.GetAncestor.restype = wt.HWND
        getter = "GetWindowLongPtrW" if ctypes.sizeof(ctypes.c_void_p) == 8 else "GetWindowLongW"
        setter = "SetWindowLongPtrW" if ctypes.sizeof(ctypes.c_void_p) == 8 else "SetWindowLongW"
        self.get_style = getattr(self.user, getter)
        self.get_style.argtypes = [wt.HWND, ctypes.c_int]
        self.get_style.restype = ctypes.c_ssize_t
        self.set_style = getattr(self.user, setter)
        self.set_style.argtypes = [wt.HWND, ctypes.c_int, ctypes.c_ssize_t]
        self.set_style.restype = ctypes.c_ssize_t
        self.user.SetWindowPos.argtypes = [
            wt.HWND,
            wt.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wt.UINT,
        ]
        self.user.SetWindowPos.restype = wt.BOOL
        self.user.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
        self.user.GetClientRect.restype = wt.BOOL
        self.user.GetDpiForWindow.argtypes = [wt.HWND]
        self.user.GetDpiForWindow.restype = wt.UINT
        self.user.SetWindowDisplayAffinity.argtypes = [wt.HWND, wt.DWORD]
        self.user.SetWindowDisplayAffinity.restype = wt.BOOL
        self.user.GetWindowDisplayAffinity.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
        self.user.GetWindowDisplayAffinity.restype = wt.BOOL
        self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wt.BOOL, wt.LPCWSTR]
        self.kernel.CreateMutexW.restype = wt.HANDLE
        self.kernel.CloseHandle.argtypes = [wt.HANDLE]
        self.kernel.CloseHandle.restype = wt.BOOL
        self.mutex = self.kernel.CreateMutexW(None, False, "Local\\AngleOverlay")
        if not self.mutex:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:
            self.kernel.CloseHandle(self.mutex)
            self.mutex = None
            raise RuntimeError(
                "Angle Overlay is already running. Open its tray icon to show settings."
            )
        monitor = primary_monitor()
        self.width, self.height = monitor.width, monitor.height

    def cursor(self):
        point = wt.POINT()
        if not self.user.GetPhysicalCursorPos(ctypes.byref(point)):
            raise ctypes.WinError(ctypes.get_last_error())
        return point.x, point.y

    def key(self, key):
        return bool(self.user.GetAsyncKeyState(key) & 0x8000)

    def topmost(self, hwnd):
        if not self.user.SetWindowPos(hwnd, wt.HWND(-1), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.mutex:
            self.kernel.CloseHandle(self.mutex)
            self.mutex = None

    def place(self, hwnd, monitor: Monitor):
        if not self.user.SetWindowPos(
            hwnd, wt.HWND(-1), monitor.left, monitor.top, monitor.width, monitor.height, 0x0010
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        bounds = wt.RECT()
        if not self.user.GetClientRect(hwnd, ctypes.byref(bounds)):
            raise ctypes.WinError(ctypes.get_last_error())
        if (bounds.right, bounds.bottom) != (monitor.width, monitor.height):
            raise RuntimeError("Overlay dimensions differ from the display's physical pixels")


class Overlay:
    def __init__(self, model, size, win, args, parent=None, on_close=None):
        self.model, self.size, self.win, self.args = model, size, win, args
        self.parent = parent
        self.on_close = on_close
        self.finished = False
        physical_dpi_context()
        self.monitor = getattr(args, "monitor", None) or primary_monitor()
        self.width, self.height = self.monitor.width, self.monitor.height
        self.root = tk.Toplevel(parent) if parent is not None else tk.Tk()
        self.root.withdraw()
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.width}x{self.height}+0+0")
        self.root.configure(background=TRANSPARENT)
        self.root.attributes("-transparentcolor", TRANSPARENT)
        self.canvas = tk.Canvas(self.root, bg=TRANSPARENT, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.line = self.canvas.create_line(
            0,
            0,
            0,
            0,
            fill=line_color(getattr(args, "line_color", "#ff0000")),
            width=args.line_width,
            state="hidden",
        )
        self.root.update_idletasks()
        self.hwnd = win.user.GetAncestor(self.root.winfo_id(), 2)
        win.set_style(self.hwnd, -20, win.get_style(self.hwnd, -20) | STYLE_MASK)
        if (win.get_style(self.hwnd, -20) & STYLE_MASK) != STYLE_MASK:
            raise RuntimeError("Windows did not accept click-through/no-activation styles")
        applied = bool(win.user.SetWindowDisplayAffinity(self.hwnd, 0x11))
        affinity = wt.DWORD(0)
        queried = bool(win.user.GetWindowDisplayAffinity(self.hwnd, ctypes.byref(affinity)))
        self.exclude_capture = applied and queried and affinity.value == 0x11
        self.root.deiconify()
        win.place(self.hwnd, self.monitor)
        self.frames = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.paused = False
        self.generation = 0
        self.latest = None
        self.previous_keys = {"pause": False, "quit": False}
        self.started = time.monotonic()
        self.next_capture = 0.0
        self.next_console = 0.0
        self.count = 0
        self.drawn = 0
        self.error = None
        self.capture_times = deque(maxlen=1000)
        self.inference_times = deque(maxlen=1000)
        self.report = {
            "device": model.device_label,
            "screen_size": [self.width, self.height],
            "monitor_device": self.monitor.device,
            "monitor_origin": [self.monitor.left, self.monitor.top],
            "monitor_dpi": [self.monitor.dpi_x, self.monitor.dpi_y],
            "input_size": list(size),
            "extended_styles": hex(win.get_style(self.hwnd, -20)),
            "click_through": True,
            "no_activate": True,
            "topmost": bool(win.get_style(self.hwnd, -20) & 0x8),
            "capture_affinity_value": affinity.value,
            "capture_exclusion_affinity": self.exclude_capture,
            "capture_backend": args.capture if self.exclude_capture else "pillow-hide",
            "capture_fallback": "hide during grab" if not self.exclude_capture else None,
        }
        self.worker = threading.Thread(target=self.infer, daemon=True)
        self.worker.start()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        print(
            f"{model.device_label} ready. "
            "Ctrl+Alt+P: pause/resume. Ctrl+Alt+Q: quit. No screenshots are saved.",
            flush=True,
        )
        self.tick_job = self.root.after(5, self.tick)

    def infer(self):
        capture = None
        try:
            physical_dpi_context()
            if self.exclude_capture:
                capture = ScreenCapture(
                    self.args.capture, self.width, self.height, monitor=self.monitor
                )
                self.report["capture_backend"] = capture.backend
            while not self.stop.is_set():
                cycle = time.monotonic()
                if capture is not None:
                    if self.paused:
                        self.stop.wait(0.02)
                        continue
                    generation = self.generation
                    point = self.win.cursor()
                    if not self.monitor.contains(*point):
                        self.stop.wait(0.02)
                        continue
                    capture_started = time.monotonic()
                    local_x, local_y = self.monitor.local_point(*point)
                    pixels = capture.crop(local_x, local_y, self.size)
                    captured = time.monotonic()
                    if pixels is None:
                        self.stop.wait(0.002)
                        continue
                    self.capture_times.append((captured - capture_started) * 1000)
                else:
                    try:
                        pixels, point, captured, generation = self.frames.get(timeout=0.2)
                    except queue.Empty:
                        continue
                started = time.monotonic()
                angle, vector = self.model.predict(pixels)
                duration = time.monotonic() - started
                self.inference_times.append(duration * 1000)
                with suppress(queue.Empty):
                    self.results.get_nowait()
                self.results.put((angle, vector, point, captured, generation, duration))
                self.stop.wait(max(0, 1 / self.args.fps - (time.monotonic() - cycle)))
        except Exception as error:
            with suppress(queue.Empty):
                self.results.get_nowait()
            self.results.put(error)
        finally:
            if capture is not None:
                self.report["roi_cache_fallbacks"] = capture.fallbacks
                capture.close()

    def tick(self):
        try:
            now = time.monotonic()
            chord = (
                not getattr(self.args, "managed_hotkeys", False)
                and self.win.key(0x11)
                and self.win.key(0x12)
            )
            for name, code in (("quit", 0x51), ("pause", 0x50)):
                pressed = chord and self.win.key(code)
                if pressed and not self.previous_keys[name]:
                    if name == "quit":
                        self.close()
                        return
                    self.paused = not self.paused
                    self.generation += 1
                    self.latest = None
                    print("Paused" if self.paused else "Resumed", flush=True)
                self.previous_keys[name] = pressed
            while True:
                try:
                    result = self.results.get_nowait()
                except queue.Empty:
                    break
                if isinstance(result, Exception):
                    raise result
                if result[4] == self.generation and not self.paused:
                    self.latest = result
                    self.count += 1
            x, y = self.win.cursor()
            inside = self.monitor.contains(x, y)
            local_x, local_y = self.monitor.local_point(x, y)
            if not self.exclude_capture and not self.paused and inside and now >= self.next_capture:
                if not self.exclude_capture:
                    self.root.withdraw()
                    self.root.update_idletasks()
                try:
                    pixels = capture_crop(
                        local_x, local_y, self.size, self.width, self.height, self.monitor
                    )
                finally:
                    if not self.exclude_capture:
                        self.root.deiconify()
                        self.win.topmost(self.hwnd)
                with contextlib.suppress(queue.Empty):
                    self.frames.get_nowait()
                self.frames.put_nowait((pixels, (x, y), time.monotonic(), self.generation))
                self.next_capture = now + 1 / self.args.fps
            visible = False
            if self.latest is not None and not self.paused and inside:
                angle, vector, point, captured, generation, duration = self.latest
                visible = math.hypot(x - point[0], y - point[1]) <= 8 and now - captured <= max(
                    2.0, 3 * duration
                )
                if visible:
                    self.canvas.coords(
                        self.line, *line_endpoints(local_x, local_y, angle, self.width, self.height)
                    )
                    self.drawn += 1
                    if now >= self.next_console:
                        print(
                            f"Angle {angle:.5f} deg | {self.report['device']} "
                            f"{duration * 1000:.0f} ms",
                            flush=True,
                        )
                        self.next_console = now + 2
            self.canvas.itemconfigure(self.line, state="normal" if visible else "hidden")
            if self.args.smoke_seconds and now - self.started >= self.args.smoke_seconds:
                self.close()
                return
            self.tick_job = self.root.after(5, self.tick)
        except Exception as error:
            self.error = str(error)
            print(f"Overlay error: {error}", file=sys.stderr, flush=True)
            self.close()

    def close(self):
        self.stop.set()
        if self.parent is None:
            self.root.quit()
        else:
            self.finish()

    def set_line_style(self, color: str, thickness: int):
        self.canvas.itemconfigure(self.line, fill=line_color(color), width=thickness)

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.stop.set()
        with suppress(tk.TclError):
            self.root.after_cancel(self.tick_job)
        self.root.destroy()
        self.worker.join(timeout=10)
        if self.worker.is_alive():
            self.error = "The inference worker did not stop within10 seconds"
        else:
            self.model.close()
        self.report.update(
            {"predictions": self.count, "drawn_ticks": self.drawn, "error": self.error}
        )
        if self.args.report:
            Path(self.args.report).write_text(json.dumps(self.report, indent=2), encoding="utf-8")
        if self.on_close is not None:
            self.on_close(self.error)

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.finish()
        if self.error:
            raise RuntimeError(self.error)


def self_test():
    for width, height in ((1920, 1080), (2560, 1440)):
        for x, y in ((0, 0), (width - 1, height - 1), (width // 2, height // 2), (3, height - 2)):
            for angle in (0, 45, 90, 135, 179.999):
                endpoints = line_endpoints(x, y, angle, width, height)
                assert endpoints is not None
                for px, py in (endpoints[:2], endpoints[2:]):
                    assert -1e-7 <= px <= width - 1 + 1e-7 and -1e-7 <= py <= height - 1 + 1e-7
                    assert min(abs(px), abs(py), abs(px - width + 1), abs(py - height + 1)) < 1e-7
                    assert (
                        abs(
                            (px - x) * math.sin(math.radians(angle))
                            - (py - y) * math.cos(math.radians(angle))
                        )
                        < 1e-7
                    )
        for x, y in ((0, 0), (width - 1, height - 1), (128, 128)):
            box, pads = crop_bounds(x, y, (256, 256), width, height)
            pixels = np.zeros((box[3] - box[1], box[2] - box[0], 3), dtype=np.uint8)
            pixels[y - box[1], x - box[0]] = [7, 29, 113]
            crop = np.pad(pixels, pads, mode="edge")
            assert crop.shape == (256, 256, 3) and np.array_equal(crop[128, 128], [7, 29, 113])
    assert np.allclose(line_endpoints(50, 50, 45, 101, 101), (0, 0, 100, 100))
    print("Geometry, screen-axis convention and edge-crop tests passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Physical-pixel display overlay; Windows, click-through, no saved captures."
    )
    parser.add_argument("--models", type=Path, default=HERE / "models")
    parser.add_argument("--runtime", choices=("auto", "cpu", "rtx", "directml"), default="auto")
    parser.add_argument("--directml-device", type=int, default=0, help="DXGI adapter index")
    parser.add_argument(
        "--fps", type=float, default=120, help="Maximum screenshot rate; no inference backlog"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--capture",
        choices=("winrt", "dxgi", "mss", "pillow", "winrt-stream", "dxgi-stream"),
        default="winrt",
    )
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument(
        "--smoke-seconds", type=float, default=0, help="Automatically close after this many seconds"
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if (
        not 0 < args.fps <= 240
        or args.threads < 1
        or args.line_width < 1
        or args.smoke_seconds < 0
        or args.directml_device < 0
    ):
        parser.error("Invalid fps, threads, line width or duration")
    if args.self_test:
        self_test()
        return
    win = Windows()
    try:
        print(f"Loading model on {args.runtime}...", flush=True)
        model = load_predictor(args.models, args.runtime, args.directml_device)
        Overlay(model, (256, 256), win, args).run()
    finally:
        win.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
