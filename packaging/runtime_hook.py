import os
import sys
from pathlib import Path

if sys.platform == "win32" and getattr(sys, "frozen", False):
    bundle = Path(sys._MEIPASS)
    # Retain handles or Windows removes the DLL search paths
    sys._angle_dll_handles = [os.add_dll_directory(str(bundle))]
