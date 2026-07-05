import sys, shutil
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]

if sys.platform.startswith("win"):
    VENV = APP_ROOT / "venv" if (APP_ROOT / "venv").exists() else APP_ROOT / ".venv"
    PYTHON   = VENV / "Scripts" / "python.exe"
    YOLO_EXE = VENV / "Scripts" / "yolo.exe"
    LABELME  = VENV / "Scripts" / "labelme.exe"
    WSL_ROOT = "/home/swt-hailo/venv_hailo"
else:  # Linux / container
    _bin = Path(sys.executable).parent          # /usr/local/bin in the image
    PYTHON   = Path(sys.executable)
    YOLO_EXE = (_bin / "yolo")    if (_bin / "yolo").exists()    else Path(shutil.which("yolo")    or "yolo")
    LABELME  = (_bin / "labelme") if (_bin / "labelme").exists() else Path(shutil.which("labelme") or "labelme")
    WSL_ROOT = None

HAILO_SCRIPT = APP_ROOT / "Scripts" / "HailoDetectionYolo.py"
MODELS  = APP_ROOT / "Models"
SCRIPTS = APP_ROOT / "Scripts"
CONFIG  = APP_ROOT / "config.json"
