from pathlib import Path
import importlib.metadata
import os
import PyInstaller.building.build_main as build_main
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
# DLL discovery must import ORT before WinRT and TensorRT
original_dependencies = build_main.find_binary_dependencies
def ort_first(binaries, import_packages, symlink_suppression_patterns):
    ordered = sorted(import_packages, key=lambda name: not name.startswith("onnxruntime"))
    return original_dependencies(binaries, ordered, symlink_suppression_patterns)
build_main.find_binary_dependencies = ort_first
datas, binaries, hidden = [], [], []
for package in ("customtkinter", "dxcam", "winrt", "tensorrt_rtx", "tensorrt_rtx_bindings", "tensorrt_rtx_libs"):
    package_data, package_binaries, package_hidden = collect_all(package)
    datas.extend(package_data)
    binaries.extend(package_binaries)
    hidden.extend(package_hidden)
hidden.extend(collect_submodules("cuda.bindings"))
hidden.extend(collect_submodules("cuda.pathfinder"))
hidden.append("pystray._win32")
for package in ("cuda-bindings", "cuda-pathfinder", "tensorrt-rtx", "onnxruntime-directml", "dxcam"):
    datas.extend(copy_metadata(package))
runtime_dist = importlib.metadata.distribution("nvidia-cuda-runtime")
cudart = [Path(runtime_dist.locate_file(p)) for p in runtime_dist.files if str(p).endswith("cudart64_13.dll")]
assert len(cudart) == 1, "Install the pinned nvidia-cuda-runtime build dependency"
binaries.append((str(cudart[0]), "."))
a = Analysis(
    [str(root / "app.py")],
    pathex=[str(root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[str(root / "packaging/runtime_hook.py")],
    excludes=["torch", "torchvision", "timm", "onnx", "onnxscript", "angle_overlay.torch_model"],
    noarchive=False,
    # Ship the LGPL tray library as replaceable Python source
    module_collection_mode={"pystray": "py"},
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="AngleOverlay", debug=False, strip=False, upx=False,
    icon=str(root / "assets/angle-overlay.ico"),
    manifest=str(root / "packaging/angle-overlay.manifest"),
    console=os.environ.get("ANGLE_OVERLAY_BUILD_CONSOLE") == "1",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="AngleOverlay")
