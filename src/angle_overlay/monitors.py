from __future__ import annotations

import ctypes
from ctypes import wintypes as wt
from dataclasses import dataclass


@dataclass(frozen=True)
class Monitor:
    device: str
    left: int
    top: int
    width: int
    height: int
    primary: bool = False
    dpi_x: int = 96
    dpi_y: int = 96

    @property
    def label(self) -> str:
        name = self.device.removeprefix("\\\\.\\")
        suffix = " · Primary" if self.primary else ""
        return f"{name} · {self.width}×{self.height} · {round(self.dpi_x / 96 * 100)}%{suffix}"

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.left + self.width and self.top <= y < self.top + self.height

    def local_point(self, x: int, y: int) -> tuple[int, int]:
        return x - self.left, y - self.top

    def desktop_box(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        left, top, right, bottom = box
        return left + self.left, top + self.top, right + self.left, bottom + self.top


def physical_dpi_context() -> None:
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    user.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    if not user.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4)):
        raise ctypes.WinError(ctypes.get_last_error())


def list_monitors() -> list[Monitor]:
    physical_dpi_context()
    user = ctypes.WinDLL("user32", use_last_error=True)
    shcore = ctypes.WinDLL("shcore")

    class MonitorInfo(ctypes.Structure):
        _fields_ = [
            ("size", wt.DWORD),
            ("monitor", wt.RECT),
            ("work", wt.RECT),
            ("flags", wt.DWORD),
            ("device", wt.WCHAR * 32),
        ]

    callback_type = ctypes.WINFUNCTYPE(
        wt.BOOL, wt.HMONITOR, wt.HDC, ctypes.POINTER(wt.RECT), wt.LPARAM
    )
    user.GetMonitorInfoW.argtypes = [wt.HMONITOR, ctypes.POINTER(MonitorInfo)]
    user.GetMonitorInfoW.restype = wt.BOOL
    user.EnumDisplayMonitors.argtypes = [wt.HDC, ctypes.c_void_p, callback_type, wt.LPARAM]
    user.EnumDisplayMonitors.restype = wt.BOOL
    shcore.GetDpiForMonitor.argtypes = [
        wt.HMONITOR,
        ctypes.c_int,
        ctypes.POINTER(wt.UINT),
        ctypes.POINTER(wt.UINT),
    ]
    shcore.GetDpiForMonitor.restype = ctypes.c_long
    monitors, failures = [], []

    @callback_type
    def collect(handle, _dc, _rect, _data):
        info = MonitorInfo()
        info.size = ctypes.sizeof(info)
        if not user.GetMonitorInfoW(handle, ctypes.byref(info)):
            failures.append(ctypes.get_last_error())
            return False
        dpi_x, dpi_y = wt.UINT(96), wt.UINT(96)
        if shcore.GetDpiForMonitor(handle, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y)) != 0:
            dpi_x.value = dpi_y.value = 96
        rect = info.monitor
        if rect.right > rect.left and rect.bottom > rect.top:
            monitors.append(
                Monitor(
                    info.device,
                    rect.left,
                    rect.top,
                    rect.right - rect.left,
                    rect.bottom - rect.top,
                    bool(info.flags & 1),
                    dpi_x.value,
                    dpi_y.value,
                )
            )
        return True

    if not user.EnumDisplayMonitors(None, None, collect, 0) or failures:
        raise ctypes.WinError(failures[0] if failures else ctypes.get_last_error())
    if not monitors:
        raise RuntimeError("Windows did not report any connected displays")
    return sorted(monitors, key=lambda monitor: (not monitor.primary, monitor.device))


def primary_monitor() -> Monitor:
    return next(monitor for monitor in list_monitors() if monitor.primary)
