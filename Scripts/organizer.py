from app_state import AppState
from paths import CONFIG
from PySide6.QtGui import QPalette, QColor, QFont
from PySide6.QtCore import QSize, Qt, QObject, Signal, QThread
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QSlider,
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

class OrganizerWindow(QDialog):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.setWindowTitle("Dataset Organizer")
        self.setMinimumSize(QSize(500, 150))

        layout = QGridLayout()
        frame = QFrame()
        main_layout = QVBoxLayout()
        
        self.path_label = QLabel("Dataset Folder Path:")
        self.path_input = QLineEdit()
        self.browse_button = QPushButton("Browse")  

        layout.addWidget(self.path_label, 0, 0)
        layout.addWidget(self.path_input, 0, 1, 1, 3)
        layout.addWidget(self.browse_button, 0, 4)
        
        self.output_label = QLabel("Output Folder Path:")
        self.output_input = QLineEdit()
        self.browse_out_button = QPushButton("Browse")  

        layout.addWidget(self.output_label, 1, 0)
        layout.addWidget(self.output_input, 1, 1, 1, 3)
        layout.addWidget(self.browse_out_button, 1, 4)
        
        self.train_val_split_label = QLabel("Train/Validation Split (% for Training):")
        self.train_val_split_input = QSlider(Qt.Horizontal)
        self.train_val_split_input.setMinimum(50)
        self.train_val_split_input.setMaximum(90)
        self.train_val_split_input.setValue(70)
        self.train_val_split_value_label = QLabel("70%")
        

        layout.addWidget(self.train_val_split_label, 2, 0, 1, 2)
        layout.addWidget(self.train_val_split_input, 2, 2, 1, 2)
        layout.addWidget(self.train_val_split_value_label, 2, 4)
        
        frame.setLayout(layout)
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        
        self.organize_button = QPushButton("Organize Dataset")
        main_layout.addWidget(frame)
        main_layout.addWidget(self.organize_button)
        self.setLayout(main_layout)
        # Connect signals to slots
        self.browse_button.clicked.connect(self.open_path_dialog)
        self.browse_out_button.clicked.connect(self.open_output_path_dialog)
        self.train_val_split_input.valueChanged.connect(self.update_split_label)
        self.organize_button.clicked.connect(self.on_organize_clicked)
        
    def open_path_dialog(self):
        # Use the static method getOpenFileName
        # Arguments: parent, title, default directory, file filters
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Directory",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks, # type: ignore
            
        )

        if path:
            self.path_input.setText(f"{path}")  
    
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
            
    def update_split_label(self, value):
        self.train_val_split_value_label.setText(f"{value}%")
       
    def on_organize_clicked(self):
        dataset_path = self.path_input.text().strip()
        output_path = self.output_input.text().strip()
        train_ratio = self.train_val_split_input.value() / 100.0

        if not dataset_path or not output_path:
            QMessageBox.warning(self, "Input Error", "Please provide both dataset and output folder paths.")
            return

        self.organize_button.setEnabled(False)

        self.thread = QThread()
        self.worker = OrganizerWorker(
            self.organize_yolo_dataset,
            dataset_path,
            output_path,
            train_ratio
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_organize_finished)
        self.worker.error.connect(self.on_organize_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def organize_yolo_dataset(
        self,
        parent_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        seed: int = 42
    ):
        parent_dir = Path(parent_dir)
        output_dir = Path(output_dir) / "YOLO_Dataset"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        assert parent_dir.exists(), "Parent directory does not exist"
        assert 0.0 < train_ratio < 1.0, "Train ratio must be between 0 and 1"

        random.seed(seed)

        # Output structure
        img_train = output_dir / "images/train"
        img_val = output_dir / "images/val"
        lbl_train = output_dir / "labels/train"
        lbl_val = output_dir / "labels/val"

        for d in [img_train, img_val, lbl_train, lbl_val]:
            d.mkdir(parents=True, exist_ok=True)

        # Step 1: discover classes
        classes = []

        for d in parent_dir.iterdir():
            if not d.is_dir():
                continue

            if d.name == "YOLO_Dataset":
                continue

            has_valid_sample = False

            for root, _, files in os.walk(d):
                for f in files:
                    img_path = Path(root) / f
                    if img_path.suffix.lower() not in IMAGE_EXTS:
                        continue

                    if img_path.with_suffix(".txt").exists() or img_path.with_suffix(".json").exists():
                        has_valid_sample = True
                        break

                if has_valid_sample:
                    break

            if has_valid_sample:
                classes.append(d.name)

        classes.sort()
        class_to_id = {name: idx for idx, name in enumerate(classes)}

        samples = []

        # Step 2: collect image + label pairs
        for class_name in classes:
            class_dir = parent_dir / class_name

            for root, _, files in os.walk(class_dir):
                root = Path(root)

                for f in files:
                    img_path = root / f
                    if img_path.suffix.lower() not in IMAGE_EXTS:
                        continue

                    label_json = img_path.with_suffix(".json")
                    label_txt = img_path.with_suffix(".txt")

                    if label_txt.exists():
                        samples.append((img_path, label_txt))
                    elif label_json.exists():
                        samples.append((img_path, label_json))
                    else:
                        # No label → skip
                        continue

        if not samples:
            raise RuntimeError("No labeled samples found")

        # Step 3: shuffle & split
        random.shuffle(samples)
        split_idx = int(len(samples) * train_ratio)

        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        # Step 4: process samples
        for split_name, split_samples in [
            ("train", train_samples),
            ("val", val_samples),
        ]:
            for img_path, label_path in split_samples:
                img_dest = (img_train if split_name == "train" else img_val) / img_path.name
                lbl_dest = (lbl_train if split_name == "train" else lbl_val) / (img_path.stem + ".txt")

                shutil.copy2(img_path, img_dest)

                if label_path.suffix == ".txt":
                    shutil.copy2(label_path, lbl_dest)
                else:
                    self.convert_labelme_to_yolo(
                        label_path,
                        lbl_dest,
                        class_to_id,
                        img_path
                    )

        # Step 5: create data.yaml
        self.create_data_yaml(output_dir, classes)

        return {
            "num_classes": len(classes),
            "num_images": len(samples),
            "train": len(train_samples),
            "val": len(val_samples),
        }
        
    def convert_labelme_to_yolo(
        self,
        labelme_json: Path,
        output_txt: Path,
        class_to_id: dict,
        image_path: Path
    ):
        with open(labelme_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        img = Image.open(image_path)
        w, h = img.size

        lines = []

        for shape in data.get("shapes", []):
            label = shape["label"]

            if label not in class_to_id:
                continue

            points = shape["points"]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            # YOLO normalized format
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h

            cls_id = class_to_id[label]

            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        output_txt.write_text("\n".join(lines))
        
    def create_data_yaml(self, output_dir: Path, classes: list):
        yaml_path = output_dir / "dataset.yaml"

        lines = [
            f"path: {output_dir.resolve()}",
            "train: images/train",
            "val: images/val",
            "",
            "names:"
        ]

        for i, name in enumerate(classes):
            lines.append(f"  {i}: {name}")

        yaml_path.write_text("\n".join(lines), encoding="utf-8")

    
    def on_organize_finished(self, result):
        self.organize_button.setEnabled(True)

        QMessageBox.information(
            self,
            "Success",
            f"Dataset organized successfully!\n\n"
            f"Classes: {result['num_classes']}\n"
            f"Total images: {result['num_images']}\n"
            f"Train: {result['train']}\n"
            f"Val: {result['val']}"
        )
        
    def on_organize_error(self, message):
        self.organize_button.setEnabled(True)
        QMessageBox.critical(self, "Error", message)


class OrganizerWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, organizer_fn, parent_dir, output_dir, train_ratio):
        super().__init__()
        self.organizer_fn = organizer_fn
        self.parent_dir = parent_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio

    def run(self):
        try:
            result = self.organizer_fn(
                parent_dir=self.parent_dir,
                output_dir=self.output_dir,
                train_ratio=self.train_ratio,
                seed=42
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app_state = AppState(str(CONFIG))
    font = QFont("Segoe UI", 11)  # Windows-safe
    app.setFont(font)
    QApplication.setStyle("Fusion")
    window = OrganizerWindow(app_state)
    window.show()
    sys.exit(app.exec())