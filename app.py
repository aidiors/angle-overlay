# ORT must load before WinRT and TensorRT to avoid DLL initialization failures
import onnxruntime

from angle_overlay.launcher import main

if __name__ == "__main__":
    onnxruntime.get_available_providers()
    main()
