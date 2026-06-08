import sys
from pathlib import Path


def is_frozen():
    return getattr(sys, "frozen", False)


if is_frozen():
    APP_ROOT     = Path(sys.executable).resolve().parents[1]
    VENV         = APP_ROOT / "venv"
    PYTHON       = VENV / "Scripts" / "python.exe"
    YOLO_EXE     = VENV / "Scripts" / "yolo.exe"
    LABELME      = VENV / "Scripts" / "labelme.exe"
    WSL_ROOT     = "/home/swt-hailo/venv_hailo"
    HAILO_SCRIPT = APP_ROOT / "Scripts" / "HailoDetectionYolo.py"

else:
    APP_ROOT = Path(__file__).resolve().parents[1]

    # Support both 'venv' (installer) and '.venv' (dev environment)
    VENV = APP_ROOT / "venv" if (APP_ROOT / "venv").exists() else APP_ROOT / ".venv"

    PYTHON       = VENV / "Scripts" / "python.exe"
    YOLO_EXE     = VENV / "Scripts" / "yolo.exe"
    LABELME      = VENV / "Scripts" / "labelme.exe"
    WSL_ROOT     = "/home/swt-hailo/venv_hailo"
    HAILO_SCRIPT = APP_ROOT / "Scripts" / "HailoDetectionYolo.py"


# Models live at APP_ROOT/Models in both installed and dev layouts
MODELS  = APP_ROOT / "Models"
SCRIPTS = APP_ROOT / "Scripts"
CONFIG  = APP_ROOT / "config.json"