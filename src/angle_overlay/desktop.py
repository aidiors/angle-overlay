from __future__ import annotations

import ctypes
import logging
import threading
from collections.abc import Callable
from ctypes import wintypes as wt
from pathlib import Path

from PIL import Image

SHORTCUTS = {1: ("toggle", 0x4F), 2: ("pause", 0x50), 3: ("stop", 0x51)}


class Hotkeys:
    def __init__(self, dispatch: Callable[[str], None]):
        self.dispatch = dispatch
        self.ready = threading.Event()
        self.errors: list[str] = []
        self.thread_id = 0
        self.thread = threading.Thread(target=self.run, name="global-hotkeys", daemon=True)
        self.thread.start()
        if not self.ready.wait(5):
            self.errors.append("Global shortcut registration timed out")

    def run(self):
        user = ctypes.WinDLL("user32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentThreadId.restype = wt.DWORD
        user.RegisterHotKey.argtypes = [wt.HWND, ctypes.c_int, wt.UINT, wt.UINT]
        user.RegisterHotKey.restype = wt.BOOL
        user.UnregisterHotKey.argtypes = [wt.HWND, ctypes.c_int]
        user.GetMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT]
        user.GetMessageW.restype = ctypes.c_int
        user.PeekMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT]
        self.thread_id = kernel.GetCurrentThreadId()
        message = wt.MSG()
        user.PeekMessageW(ctypes.byref(message), None, 0, 0, 0)
        registered = []
        try:
            for identifier, (_, key) in SHORTCUTS.items():
                if user.RegisterHotKey(None, identifier, 0x4003, key):
                    registered.append(identifier)
                else:
                    self.errors.append(f"Ctrl+Alt+{chr(key)} is already in use or unavailable")
            self.ready.set()
            while True:
                result = user.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result <= 0:
                    if result == -1:
                        raise ctypes.WinError(ctypes.get_last_error())
                    break
                if message.message == 0x0312 and message.wParam in SHORTCUTS:
                    self.dispatch(SHORTCUTS[message.wParam][0])
        except Exception:
            logging.exception("Global hotkey service failed")
            self.dispatch("hotkey_error")
        finally:
            for identifier in registered:
                user.UnregisterHotKey(None, identifier)
            self.ready.set()

    def close(self):
        if self.thread_id and self.thread.is_alive():
            user = ctypes.WinDLL("user32", use_last_error=True)
            user.PostThreadMessageW.argtypes = [wt.DWORD, wt.UINT, wt.WPARAM, wt.LPARAM]
            user.PostThreadMessageW.restype = wt.BOOL
            if not user.PostThreadMessageW(self.thread_id, 0x0012, 0, 0):
                raise ctypes.WinError(ctypes.get_last_error())
            self.thread.join(5)


class Tray:
    def __init__(self, icon_path: Path, dispatch: Callable[[str], None]):
        import pystray

        self.ready = threading.Event()
        self.available = False

        def action(name):
            def callback(_icon, _item):
                dispatch(name)

            return callback

        with Image.open(icon_path) as image:
            self.icon = pystray.Icon(
                "AngleOverlay",
                image.copy(),
                "Angle Overlay",
                menu=pystray.Menu(
                    pystray.MenuItem("Show settings", action("show"), default=True),
                    pystray.MenuItem("Toggle overlay  ·  Ctrl+Alt+O", action("toggle")),
                    pystray.MenuItem("Pause / resume  ·  Ctrl+Alt+P", action("pause")),
                    pystray.MenuItem("Stop overlay  ·  Ctrl+Alt+Q", action("stop")),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem("Exit", action("exit")),
                ),
            )

        def setup(icon):
            icon.visible = True
            self.available = True
            self.ready.set()

        def run():
            try:
                self.icon.run(setup=setup)
            except Exception:
                logging.exception("Notification icon unavailable")
                dispatch("tray_error")
            finally:
                self.available = False
                self.ready.set()

        self.thread = threading.Thread(target=run, name="tray-icon", daemon=True)
        self.thread.start()
        self.ready.wait(5)

    def close(self):
        self.icon.stop()
        self.thread.join(5)
