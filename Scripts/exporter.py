from app_state import AppState
from paths import APP_DIR, SCRIPTS, PYTHON, WSL_ROOT, HAILO_SCRIPT
from PySide6.QtGui import QPalette, QColor, QFont
from PySide6.QtCore import QSize, Qt, QObject, Signal, QThread
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QSpinBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QGridLayout,  
    QDialog, 
    QFrame,
    QVBoxLayout,
    QFileDialog,
    QMessageBox
    )
import sys
import os
import json
import random
import shutil
from pathlib import Path
from PIL import Image
import subprocess
import yaml
import zipfile

SUBPROCESS_FLAGS = 0x08000000  # CREATE_NO_WINDOW

class Exporter(QDialog):
    def __init__(self, app_state: AppState = None):
        super().__init__()
        self.setWindowTitle("Model Exporter")
        self.app_state = app_state
        self.setMinimumSize(QSize(500, 300))

        layout = QGridLayout()
        frame_top = QFrame()
        main_layout = QVBoxLayout()
        
        self.onnx_path_label = QLabel("ONNX File Path:")
        self.onnx_path_input = QLineEdit()
        self.browse_onnx_button = QPushButton("Browse")  

        layout.addWidget(self.onnx_path_label, 0, 0)
        layout.addWidget(self.onnx_path_input, 0, 1, 1, 3)
        layout.addWidget(self.browse_onnx_button, 0, 4)
        
        self.output_label = QLabel("Output Folder Path:")
        self.output_input = QLineEdit()
        self.browse_out_button = QPushButton("Browse")  

        layout.addWidget(self.output_label, 1, 0)
        layout.addWidget(self.output_input, 1, 1, 1, 3)
        layout.addWidget(self.browse_out_button, 1, 4)
        
        self.yaml_file_label = QLabel("YAML File Path:")
        self.yaml_file_input = QLineEdit()
        self.browse_yaml_button = QPushButton("Browse")

        layout.addWidget(self.yaml_file_label, 2, 0)
        layout.addWidget(self.yaml_file_input, 2, 1, 1, 3)
        layout.addWidget(self.browse_yaml_button, 2, 4)
        
        frame_bottom = QFrame()
        bottom_layout = QGridLayout()
        
        self.resolution_label = QLabel("Model Resolution:")
        self.resolution_input = QSpinBox()
        self.resolution_input.setRange(160, 2048)
        self.resolution_input.setValue(640)
        self.model_name_label = QLabel("Model Name:")
        self.model_name_input = QLineEdit()

        bottom_layout.addWidget(self.resolution_label, 0, 0)
        bottom_layout.addWidget(self.resolution_input, 0, 1)
        bottom_layout.addWidget(self.model_name_label, 1, 0)
        bottom_layout.addWidget(self.model_name_input, 1, 1)
        
        
        self.export_button = QPushButton("Export Model")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(5000)
        self.log_box.setPlaceholderText("Logs will appear here...")

        frame_bottom.setLayout(bottom_layout)
        frame_top.setLayout(layout)
        main_layout.addWidget(frame_top)
        main_layout.addWidget(frame_bottom)
        main_layout.addWidget(self.export_button)
        main_layout.addWidget(self.log_box)
        
        frame_top.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        frame_bottom.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)

        self.setLayout(main_layout)
        
        self._connect_signals() 
        
    def _connect_signals(self):
        self.browse_onnx_button.clicked.connect(self.open_onnx_path_dialog)
        self.browse_out_button.clicked.connect(self.open_output_path_dialog)
        self.browse_yaml_button.clicked.connect(self.open_yaml_dialog)
        self.export_button.clicked.connect(self.start_export)
        
    def open_onnx_path_dialog(self):
        # Use the static method getOpenFileName
        # Arguments: parent, title, default directory, file filters
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select ONNX File",
            "",
            "ONNX Files (*.onnx);;All Files (*.*)",
        )
        if filename:
            self.onnx_path_input.setText(f"{filename}")  
    
    def open_output_path_dialog(self):
        # Use the static method getOpenFileName
        # Arguments: parent, title, default directory, file filters
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks, # type: ignore
            
        )

        if path:
            self.output_input.setText(f"{path}")
            
    def open_yaml_dialog(self):
        # Use the static method getOpenFileName
        # Arguments: parent, title, default directory, file filters
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select YAML File",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*.*)",
            
        )

        if filename:
            self.yaml_file_input.setText(f"{filename}") 
            
    def start_export(self):
        self.export_button.setEnabled(False)

        self.thread = QThread()
        self.worker = ExportWorker(
            self.onnx_path_input.text(),
            self.yaml_file_input.text(),
            self.output_input.text(),
            self.resolution_input.value(),
            self.model_name_input.text(),
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.log_box.appendPlainText)
        self.worker.finished.connect(self.on_export_finished)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        
    def on_export_finished(self, success, message):
        self.export_button.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)

def get_random_images_from_yaml(yaml_path, count=5):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data["path"])
    train_dir = dataset_root / data.get("train", "images/train")

    if not train_dir.exists():
        raise RuntimeError(f"Train image directory not found: {train_dir}")

    images = [
        p for p in train_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if len(images) < count:
        raise RuntimeError(
            f"Need at least {count} images for Hailo conversion, found {len(images)}"
        )

    return random.sample(images, count)


        
class ExportWorker(QObject):
    log = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, onnx_path, yaml_path, output_path, resolution, model_name):
        super().__init__()
        self.onnx_path = onnx_path
        self.yaml_path = yaml_path
        self.output_path = output_path
        self.resolution = resolution
        self.model_name = model_name
        
    def convert_onnx_to_hef(self, onnx_path: Path, resolution: int, classes: int) -> Path:
        py = f"{WSL_ROOT}/bin/python"

        scripts = {
            "cleanup": f"{py} {WSL_ROOT}/cleanup.py",
            "parse": f"{py} {WSL_ROOT}/parse.py --width {resolution} --height {resolution}",
            "optimize": f"{py} {WSL_ROOT}/optimize.py --width {resolution} --height {resolution} --resize_side {resolution}",
            "compile": f"{py} {WSL_ROOT}/compile.py",
        }

        self.log.emit("Copying ONNX to WSL...")
        onnx_wsl = self.to_wsl_path(onnx_path)
        wsl_target = f"{WSL_ROOT}/{onnx_path.name}"

        subprocess.run(
            ["wsl", "cp", "-f", onnx_wsl, wsl_target],
            check=True, creationflags=SUBPROCESS_FLAGS
        )

        self.log.emit("Cleaning WSL workspace...")
        subprocess.run(
            ["wsl", "bash", "-c", scripts["cleanup"]],
            check=True, creationflags=SUBPROCESS_FLAGS
        )

        # -------------------------
        # Copy 5 random images (from YAML)
        # -------------------------
        self.log.emit("Copying calibration images...")

        images = get_random_images_from_yaml(self.yaml_path, count=5)

        for i, img in enumerate(images, 1):
            self.log.emit(f"Copying image {i}/5: {img.name}")
            img_wsl = self.to_wsl_path(img)
            subprocess.run(
                ["wsl", "cp", img_wsl, WSL_ROOT],
                check=True, creationflags=SUBPROCESS_FLAGS
            )


        self.log.emit("Updating Hailo config...")
        json_cfg = f"{WSL_ROOT}/yolov8n_nms_config.json"

        read = subprocess.run(
            ["wsl", "cat", json_cfg],
            capture_output=True, text=True, check=True
        )

        cfg = json.loads(read.stdout)
        cfg["classes"] = classes
        cfg["image_dims"] = [resolution, resolution]

        subprocess.run(
            ["wsl", "tee", json_cfg],
            input=json.dumps(cfg, indent=2),
            text=True, check=True
        )

        for step in ("parse", "optimize", "compile"):
            self.log.emit(f"{step.capitalize()} step...")
            subprocess.run(
                ["wsl", "bash", "-c", scripts[step]],
                check=True, creationflags=SUBPROCESS_FLAGS
            )

        self.log.emit("Retrieving HEF...")
        find = subprocess.run(
            ["wsl", "bash", "-c", f"find {WSL_ROOT} -maxdepth 1 -name '*.hef'"],
            capture_output=True, text=True, check=True
        )

        hef_files = [f for f in find.stdout.splitlines() if f.strip()]
        if not hef_files:
            raise RuntimeError("HEF file was not generated")

        hef_wsl = hef_files[0]
        hef_name = Path(hef_wsl).name
        hef_out = Path(self.output_path) / hef_name

        subprocess.run(
            ["wsl", "cp", hef_wsl, self.to_wsl_path(hef_out)],
            check=True, creationflags=SUBPROCESS_FLAGS
        )

        return hef_out


    def run(self):
        onnx_path = Path(self.onnx_path).expanduser().resolve()
        
        subprocess.run(
            ["wsl", "echo", "WSL OK"],
            check=True,
            creationflags=SUBPROCESS_FLAGS
        )

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found:\n{onnx_path}")

        try:
            self.log.emit("Starting export...")

            onnx_path = Path(self.onnx_path)
            output_dir = Path(self.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # -------------------------
            # Step 1: Parse YAML
            # -------------------------
            self.log.emit("Parsing YAML...")
            num_classes, class_names = self.parse_yaml(self.yaml_path)

            # -------------------------
            # Step 2: ONNX -> HEF
            # -------------------------
            hef_path = self.convert_onnx_to_hef(
                onnx_path,
                self.resolution,
                num_classes
            )

            # -------------------------
            # Step 3: Write JSON files
            # -------------------------
            labels_json = output_dir / "labels.json"
            model_json = output_dir / "model.json"
        
            with open(labels_json, "w", encoding="utf-8") as f:
                json.dump(
                    {str(k): v for k, v in class_names.items()},
                    f,
                    indent=2
                )

            model_cfg = {
                "ConfigVersion": 10,
                "DEVICE": [
                    {
                        "DeviceType": "HAILO8L",
                        "RuntimeAgent": "HAILORT",
                        "SupportedDeviceTypes": "HAILORT/HAILO8L"
                    }
                ],
                "PRE_PROCESS": [
                    {
                        "InputType": "Image",
                        "ImageBackend": "opencv",
                        "InputPadMethod": "letterbox",
                        "InputResizeMethod": "bilinear",
                        "InputN": 1,
                        "InputH": self.resolution,
                        "InputW": self.resolution,
                        "InputC": 3,
                        "InputQuantEn": True
                    }
                ],
                "MODEL_PARAMETERS": [
                    {
                        "ModelPath": "model.hef"
                    }
                ],
                "POST_PROCESS": [
                    {
                        "OutputPostprocessType": "DetectionYoloV8",
                        "PythonFile": "HailoDetectionYolo.py",
                        "OutputNumClasses": num_classes,
                        "OutputClassIDAdjustment": 1,
                        "LabelsPath": "labels.json"
                    }
                ]
            }

            with open(model_json, "w", encoding="utf-8") as f:
                json.dump(model_cfg, f, indent=2)


            # -------------------------
            # Step 4: ZIP
            # -------------------------
            model_name = self.model_name.strip() or onnx_path.stem
            zip_path = output_dir / f"{model_name}.zip"

            self.log.emit("Creating ZIP package...")
            yolo_script = HAILO_SCRIPT

            if not yolo_script.exists():
                raise RuntimeError(f"Missing HailoDetectionYolo.py at {yolo_script}")



            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(hef_path, "model.hef")
                z.write(labels_json, "labels.json")
                z.write(model_json, "model.json")
                z.write(yolo_script, "HailoDetectionYolo.py")

            self.finished.emit(True, f"Export completed:\n{zip_path}")

        except Exception as e:
            self.finished.emit(False, str(e))

            
    def to_wsl_path(self, win_path) -> str:
        win_path = Path(win_path).resolve()

        # Convert to forward slashes explicitly
        win_path_str = win_path.as_posix()

        result = subprocess.run(
            ["wsl", "wslpath", "-a", win_path_str],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"WSL path conversion failed for:\n{win_path}\n\n"
                f"WSL output:\n{result.stderr.strip() or result.stdout.strip()}"
            )

        return result.stdout.strip()




    def parse_yaml(self,yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        names = data["names"]
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        nc = len(names)

        return nc, names



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app_state = AppState("config,json")
    font = QFont("Segoe UI", 11)  # Windows-safe
    app.setFont(font)
    QApplication.setStyle("Fusion")
    window = Exporter(app_state)
    window.show()
    sys.exit(app.exec())