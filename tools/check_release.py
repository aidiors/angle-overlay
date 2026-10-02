from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "angle-fp32.onnx": "angle-fp32.json",
    "angle-directml.onnx": "angle-directml.json",
    "angle-bf16.rtx": "angle-bf16.manifest.json",
}
DOCUMENTS = (
    "README.md",
    "README.ru.md",
    "USER-GUIDE.md",
    "USER-GUIDE.ru.md",
    "TECHNICAL.md",
    "TECHNICAL.ru.md",
    "LICENSE",
    "NOTICE",
)


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]["version"]


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_models(folder: Path) -> None:
    sources = set()
    for filename, manifest_name in MODELS.items():
        artifact = folder / filename
        if not artifact.is_file() or artifact.stat().st_size < 1024:
            raise RuntimeError(f"Missing model or unresolved Git LFS pointer: {artifact}")
        manifest = json.loads((folder / manifest_name).read_text(encoding="utf-8"))
        if file_hash(artifact) != manifest["files"][filename]:
            raise RuntimeError(f"Model checksum mismatch: {filename}")
        if manifest["input_shape"] != [1, 3, 256, 256]:
            raise RuntimeError(f"Unexpected input shape: {filename}")
        if manifest["input_dtype"] != "float32":
            raise RuntimeError(f"Unexpected input dtype: {filename}")
        sources.add(manifest["source_sha256"])
        if filename == "angle-directml.onnx" and manifest.get("opset") != 17:
            raise RuntimeError("DirectML requires its dedicated opset17 export")
        if filename.endswith(".rtx"):
            with (ROOT / "pyproject.toml").open("rb") as stream:
                groups = tomllib.load(stream)["dependency-groups"]
            pinned = next(
                r.split("==", 1)[1] for r in groups["rtx"] if r.startswith("tensorrt-rtx==")
            )
            if manifest["runtime_version"] != pinned:
                raise RuntimeError("RTX engine does not match the pinned runtime")
    if len(sources) != 1:
        raise RuntimeError("Deployment models have different source checkpoints")


def release_identity(tag: str) -> tuple[str, str]:
    version = project_version()
    expected = f"v{version}"
    if not re.fullmatch(r"v\d+\.\d+\.\d+", expected):
        raise ValueError("Release versions must use major.minor.patch")
    tag = tag or expected
    if tag != expected:
        raise ValueError(f"Tag {tag!r} does not match package version {expected!r}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    existing = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if existing.returncode == 0 and existing.stdout.strip() != commit:
        raise ValueError("The release tag points to another commit")
    return tag, commit


def check_bundle(folder: Path) -> None:
    folder = folder.resolve()
    executable = folder / "AngleOverlay.exe"
    for filename in (
        *DOCUMENTS,
        "assets/angle-overlay.ico",
        "assets/screenshot.png",
        "_internal/pystray/_win32.py",
    ):
        if not (folder / filename).is_file():
            raise RuntimeError(f"Missing bundle file: {filename}")
    if not executable.is_file():
        raise RuntimeError("Missing Windows executable")
    check_models(folder / "models")
    if not any((folder / "licenses").rglob("COPYING.LGPL")):
        raise RuntimeError("The tray library's LGPL notice is missing")
    if (folder / "_internal/torch").exists():
        raise RuntimeError("Developer PyTorch must not be bundled")
    with tempfile.TemporaryDirectory(prefix="angle-overlay-check-") as temporary:
        report = Path(temporary) / "cpu.json"
        environment = dict(os.environ)
        environment.update(
            PATH=f"{os.environ['SYSTEMROOT']}/System32;{os.environ['SYSTEMROOT']}",
            PYTHONPATH="",
            PYTHONHOME="",
            CUDA_PATH="",
            CUDA_HOME="",
            LOCALAPPDATA=temporary,
        )
        subprocess.run(
            [str(executable), "--self-check", "--runtime", "cpu", "--report", str(report)],
            cwd=temporary,
            env=environment,
            timeout=120,
            check=True,
        )
        payload = json.loads(report.read_text(encoding="utf-8"))
        if not payload.get("ok") or not payload["device"].startswith("CPU"):
            raise RuntimeError("Frozen CPU self-check failed")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify deployable model integrity, release identity and the frozen CPU runtime."
        )
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--tag", nargs="?", const="", default=None)
    args = parser.parse_args()
    check_models(ROOT / "models")
    if args.tag is not None:
        tag, commit = release_identity(args.tag)
        if output := os.environ.get("GITHUB_OUTPUT"):
            with Path(output).open("a", encoding="utf-8") as stream:
                stream.write(f"tag={tag}\ncommit={commit}\nversion={project_version()}\n")
        print(f"Release identity verified: {tag}")
    if args.bundle is not None:
        check_bundle(args.bundle)
        print("Frozen bundle and isolated-path CPU runtime verified")
    print("All deployment model hashes verified")


if __name__ == "__main__":
    main()
