from __future__ import annotations

import os
import sys
from pathlib import Path


def app_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def data_directory() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local/share")) / "AngleOverlay"
    root.mkdir(parents=True, exist_ok=True)
    return root
