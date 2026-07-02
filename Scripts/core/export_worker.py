"""
core/export_worker.py  —  TRAINR
Pure export backend: ONNX → HEF → ZIP via WSL + Hailo toolchain.
No Qt widgets — safe to import from any context.
"""
from __future__ import annotations

import json
import random
import subprocess
import zipfile
from pathlib import Path

import yaml
from PySide6.QtCore import QObject, Signal

from paths import WSL_ROOT, HAILO_SCRIPT

SUBPROCESS_FLAGS = 0x08000000  # CREATE_NO_WINDOW  (Windows only, ignored elsewhere)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_random_images_from_yaml(yaml_path: str, count: int = 64) -> list[Path]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data["path"])
    train_dir    = dataset_root / data.get("train", "images/train")

    if not train_dir.exists():
        raise RuntimeError(f"Train image directory not found: {train_dir}")

    images = [
        p for p in train_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if len(images) < count:
        raise RuntimeError(
            f"Need at least {count} images for Hailo conversion, "
            f"found {len(images)}"
        )

    return random.sample(images, count)


def parse_yaml(yaml_path: str) -> tuple[int, dict]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return len(names), names


def to_wsl_path(win_path) -> str:
    win_path = Path(win_path).resolve()
    result   = subprocess.run(
        ["wsl", "wslpath", "-a", win_path.as_posix()],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"WSL path conversion failed for:\n{win_path}\n\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────────

class ExportWorker(QObject):
    log      = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, onnx_path: str, yaml_path: str, output_path: str,
                 resolution: int, model_name: str):
        super().__init__()
        self.onnx_path   = onnx_path
        self.yaml_path   = yaml_path
        self.output_path = output_path
        self.resolution  = resolution
        self.model_name  = model_name

    # ── public entry ──────────────────────────────────────────────────────────

    def run(self):
        try:
            if WSL_ROOT is None:
                raise RuntimeError(
                    "HEF export via WSL is not available on this platform "
                    "(no Hailo Dataflow Compiler). Use 'Package for HEF' to "
                    "produce a bundle, then compile it on an x86 machine."
                )
            subprocess.run(
                ["wsl", "echo", "WSL OK"],
                check=True, creationflags=SUBPROCESS_FLAGS,
            )
            onnx_path  = Path(self.onnx_path).expanduser().resolve()
            output_dir = Path(self.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX file not found:\n{onnx_path}")

            self.log.emit("Starting export…")
            self.log.emit("Parsing YAML…")
            num_classes, class_names = parse_yaml(self.yaml_path)

            hef_path = self._convert_onnx_to_hef(onnx_path, self.resolution,
                                                  num_classes)

            # JSON artefacts
            labels_json = output_dir / "labels.json"
            model_json  = output_dir / "model.json"

            with open(labels_json, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in class_names.items()}, f, indent=2)

            model_cfg = {
                "ConfigVersion": 10,
                "DEVICE": [{"DeviceType": "HAILO8L",
                             "RuntimeAgent": "HAILORT",
                             "SupportedDeviceTypes": "HAILORT/HAILO8L"}],
                "PRE_PROCESS": [{"InputType": "Image",
                                  "ImageBackend": "opencv",
                                  "InputPadMethod": "letterbox",
                                  "InputResizeMethod": "bilinear",
                                  "InputN": 1,
                                  "InputH": self.resolution,
                                  "InputW": self.resolution,
                                  "InputC": 3,
                                  "InputQuantEn": True}],
                "MODEL_PARAMETERS": [{"ModelPath": "model.hef"}],
                "POST_PROCESS": [{"OutputPostprocessType": "DetectionYoloV8",
                                   "PythonFile": "HailoDetectionYolo.py",
                                   "OutputNumClasses": num_classes,
                                   "OutputClassIDAdjustment": 1,
                                   "LabelsPath": "labels.json"}],
            }

            with open(model_json, "w", encoding="utf-8") as f:
                json.dump(model_cfg, f, indent=2)

            # ZIP
            model_name = self.model_name.strip() or onnx_path.stem
            zip_path   = output_dir / f"{model_name}.zip"

            self.log.emit("Creating ZIP package…")
            if not HAILO_SCRIPT.exists():
                raise RuntimeError(
                    f"Missing HailoDetectionYolo.py at {HAILO_SCRIPT}")

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(hef_path,    "model.hef")
                z.write(labels_json, "labels.json")
                z.write(model_json,  "model.json")
                z.write(HAILO_SCRIPT, "HailoDetectionYolo.py")

            self.finished.emit(True, f"Export completed:\n{zip_path}")

        except Exception as exc:
            self.finished.emit(False, str(exc))

    # ── private ───────────────────────────────────────────────────────────────

    def _convert_onnx_to_hef(self, onnx_path: Path,
                               resolution: int, classes: int) -> Path:
        py = f"{WSL_ROOT}/bin/python"
        scripts = {
            "cleanup":  f"{py} {WSL_ROOT}/cleanup.py",
            "parse":    (f"{py} {WSL_ROOT}/parse.py "
                         f"--width {resolution} --height {resolution}"),
            "optimize": (f"{py} {WSL_ROOT}/optimize.py "
                         f"--width {resolution} --height {resolution} "
                         f"--resize_side {resolution}"),
            "compile":  f"{py} {WSL_ROOT}/compile.py",
        }

        self.log.emit("Copying ONNX to WSL…")
        onnx_wsl   = to_wsl_path(onnx_path)
        wsl_target = f"{WSL_ROOT}/{onnx_path.name}"
        subprocess.run(
            ["wsl", "cp", "-f", onnx_wsl, wsl_target],
            check=True, creationflags=SUBPROCESS_FLAGS,
        )

        self.log.emit("Cleaning WSL workspace…")
        subprocess.run(
            ["wsl", "bash", "-c", scripts["cleanup"]],
            check=True, creationflags=SUBPROCESS_FLAGS,
        )

        self.log.emit("Copying calibration images…")
        images = get_random_images_from_yaml(self.yaml_path, count=64)
        for i, img in enumerate(images, 1):
            self.log.emit(f"  Copying image {i}/64: {img.name}")
            subprocess.run(
                ["wsl", "cp", to_wsl_path(img), WSL_ROOT],
                check=True, creationflags=SUBPROCESS_FLAGS,
            )

        self.log.emit("Updating Hailo config…")
        json_cfg = f"{WSL_ROOT}/yolov8n_nms_config.json"
        read = subprocess.run(
            ["wsl", "cat", json_cfg],
            capture_output=True, text=True, check=True,
        )
        cfg = json.loads(read.stdout)
        cfg["classes"]    = classes
        cfg["image_dims"] = [resolution, resolution]
        subprocess.run(
            ["wsl", "tee", json_cfg],
            input=json.dumps(cfg, indent=2), text=True, check=True,
        )

        for step in ("parse", "optimize", "compile"):
            self.log.emit(f"{step.capitalize()} step…")
            subprocess.run(
                ["wsl", "bash", "-c", scripts[step]],
                check=True, creationflags=SUBPROCESS_FLAGS,
            )

        self.log.emit("Retrieving HEF…")
        find = subprocess.run(
            ["wsl", "bash", "-c",
             f"find {WSL_ROOT} -maxdepth 1 -name '*.hef'"],
            capture_output=True, text=True, check=True,
        )
        hef_files = [f for f in find.stdout.splitlines() if f.strip()]
        if not hef_files:
            raise RuntimeError("HEF file was not generated")

        hef_wsl  = hef_files[0]
        hef_name = Path(hef_wsl).name
        hef_out  = Path(self.output_path) / hef_name
        subprocess.run(
            ["wsl", "cp", hef_wsl, to_wsl_path(hef_out)],
            check=True, creationflags=SUBPROCESS_FLAGS,
        )
        return hef_out
