import sys
from pathlib import Path
import shutil

def is_frozen():
    return getattr(sys, "frozen", False)

if is_frozen():
    # interface.exe → <root>/app/interface.exe
    APP_ROOT = Path(sys.executable).resolve().parents[1]
    PYTHON   = APP_ROOT / "venv" / "Scripts" / "python.exe"
    YOLO_EXE    = APP_ROOT / "venv" / "Scripts" / "yolo.exe"
    LABELME = APP_ROOT / "venv" / "Scripts" / "labelme.exe"
    WSL_ROOT = "/home/swt-hailo/venv_hailo"
    HAILO_SCRIPT = APP_ROOT / "app" / "Scripts" / "HailoDetectionYolo.py"

else:
    # Dev mode (Visual Studio, python interface.py)
    APP_ROOT = Path(__file__).resolve().parents[1]

    # Use *current* python environment
    PYTHON = Path(sys.executable)

    # Try to find yolo / labelme in PATH
    YOLO_EXE = Path(shutil.which("yolo") or "")
    LABELME = Path(shutil.which("labelme") or "")
    WSL_ROOT = "/home/swt-hailo/venv_hailo"
    HAILO_SCRIPT = APP_ROOT / "Scripts" / "HailoDetectionYolo.py"

APP_DIR  = APP_ROOT / "app"
MODELS  = APP_DIR / "Models"
SCRIPTS = APP_DIR / "Scripts"
CONFIG  = APP_ROOT / "config.json"
