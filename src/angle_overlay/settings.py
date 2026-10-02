from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

TRANSPARENT = "#010203"


def line_color(value: str) -> str:
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value) or value.lower() == TRANSPARENT:
        raise ValueError(
            "Choose an RGB hex color other than the reserved transparency color #010203"
        )
    return value.lower()


@dataclass(frozen=True)
class Settings:
    backend: str = "auto"
    directml_device: int = 0
    monitor_device: str = ""
    color: str = "#ff0000"
    thickness: int = 2

    def __post_init__(self):
        if self.backend not in {"auto", "rtx", "directml", "cpu"}:
            raise ValueError("Unknown inference backend")
        if type(self.directml_device) is not int or self.directml_device < 0:
            raise ValueError("Invalid GPU adapter index")
        if type(self.thickness) is not int or not 1 <= self.thickness <= 12:
            raise ValueError("Thickness must be1..12 physical pixels")
        if not isinstance(self.monitor_device, str):
            raise ValueError("Invalid display name")
        line_color(self.color)


def load_settings(path: Path) -> Settings:
    if not path.exists():
        return Settings()
    try:
        return Settings(**json.loads(path.read_text(encoding="utf-8")))
    except OSError, ValueError, TypeError:
        logging.exception("Invalid preferences; using defaults")
        return Settings()


def save_settings(path: Path, settings: Settings) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(settings), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
