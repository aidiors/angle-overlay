from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "dist/AngleOverlay"
    if not (destination / "AngleOverlay.exe").is_file():
        raise RuntimeError("Build the executable first")
    models = destination / "models"
    models.mkdir(exist_ok=True)
    for name in (
        "angle-fp32.onnx",
        "angle-fp32.json",
        "angle-directml.onnx",
        "angle-directml.json",
        "angle-bf16.rtx",
        "angle-bf16.manifest.json",
    ):
        shutil.copy2(ROOT / "models" / name, models / name)
    for name in (
        "LICENSE",
        "NOTICE",
        "README.md",
        "README.ru.md",
        "USER-GUIDE.md",
        "USER-GUIDE.ru.md",
        "TECHNICAL.md",
        "TECHNICAL.ru.md",
    ):
        shutil.copy2(ROOT / name, destination / name)
    assets_folder = destination / "assets"
    assets_folder.mkdir(exist_ok=True)
    for name in ("angle-overlay.ico", "screenshot.png"):
        shutil.copy2(ROOT / "assets" / name, assets_folder / name)
    licenses = destination / "licenses"
    licenses.mkdir(exist_ok=True)
    inventory = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        found = []
        for relative in distribution.files or ():
            if not any(
                token in str(relative).lower() for token in ("license", "copying", "notice")
            ):
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file() or source.suffix.lower() not in {
                "",
                ".txt",
                ".md",
                ".rst",
                ".lgpl",
                ".gpl",
            }:
                continue
            target = licenses / name / str(relative).replace("../", "")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            found.append(str(target.relative_to(destination)).replace("\\", "/"))
        inventory.append({"package": name, "version": distribution.version, "notices": found})
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.exists():
        shutil.copy2(python_license, licenses / "Python-LICENSE.txt")
    (licenses / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    assets = ROOT / "release-assets"
    assets.mkdir(exist_ok=True)
    if args.zip:
        with (ROOT / "pyproject.toml").open("rb") as stream:
            version = tomllib.load(stream)["project"]["version"]
        archive = Path(
            shutil.make_archive(
                str(assets / f"AngleOverlay-{version}-windows-x64"),
                "zip",
                destination,
            )
        )
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.with_suffix(".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        print("Created release archive:", archive.name)
    else:
        print("Staged local models and notices")


if __name__ == "__main__":
    main()
