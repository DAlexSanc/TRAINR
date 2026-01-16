import sys
import os 
import subprocess
from app_state import AppState
from organizer import OrganizerWindow
from exporter import Exporter
from paths import PYTHON, YOLO_EXE, LABELME, MODELS, CONFIG
from PySide6.QtGui import QPalette, QColor, QFont
from PySide6.QtCore import QSize, Qt, QProcess
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QGridLayout,  
    QHBoxLayout, 
    QPlainTextEdit, 
    QFrame,
    QToolBar,
    QFileDialog
    )

class MainWindow(QMainWindow):
    def __init__(self, app_state=None):
        super().__init__()
        self.state = app_state
        self.current_job = None
        self._setup_window()
        self._create_widgets()
        self._build_layouts()
        self._create_toolbar()
        self._connect_signals()

        if self.state:
            self.load_state()
            self.bind_state()

    def _setup_window(self):
        self.setWindowTitle("TRAINR")
        self.resize(750, 500)
        self.setMinimumSize(750, 500)

        self.central = QFrame()
        self.central.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setCentralWidget(self.central)
        
    def _create_widgets(self):
        # Dataset
        self.dataset_label = QLabel("Path to Dataset YAML:")
        self.dataset_linedit = QLineEdit()
        self.dataset_linedit.setPlaceholderText("Path to dataset YAML")
        self.dataset_button = QPushButton("Browse")

        # Output
        self.output_label = QLabel("Output Path:")
        self.output_linedit = QLineEdit()
        self.output_linedit.setPlaceholderText("Path to output directory")
        self.output_button = QPushButton("Browse")

        # Model
        self.model_label = QLabel("Model Size to train:")
        self.model_combobox = QComboBox()
        self.model_combobox.addItems(
            ["DetectionN", "DetectionS", "DetectionM", "DetectionL", "DetectionX"]
        )

        # Training params
        self.resolution_spinbox = QSpinBox()
        self.resolution_spinbox.setRange(64, 2048)
        self.resolution_spinbox.setValue(640)

        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(100)

        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setRange(1, 100)
        self.patience_spinbox.setValue(10)

        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 1024)
        self.batch_spinbox.setValue(16)

        self.auto_batch_checkbox = QCheckBox("Use Auto Batch Size")

        # Actions
        self.train_button = QPushButton("Start Training")

        # Logs
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(5000)
        self.log_box.setPlaceholderText("Logs will appear here...")

    def _build_layouts(self):
        # Left panel
        left_layout = QGridLayout()
        left_layout.addWidget(self.dataset_label, 0, 0, 1, 2)
        left_layout.addWidget(self.dataset_linedit, 1, 0, 1, 2)
        left_layout.addWidget(self.dataset_button, 1, 2)

        left_layout.addWidget(self.output_label, 2, 0, 1, 2)
        left_layout.addWidget(self.output_linedit, 3, 0, 1, 2)
        left_layout.addWidget(self.output_button, 3, 2)

        left_layout.addWidget(self.model_label, 4, 0)
        left_layout.addWidget(self.model_combobox, 4, 1, 1, 2)

        left_frame = QFrame()
        left_frame.setLayout(left_layout)

        # Right panel
        right_layout = QGridLayout()
        right_layout.addWidget(QLabel("Image Resolution:"), 0, 0)
        right_layout.addWidget(self.resolution_spinbox, 0, 1)

        right_layout.addWidget(QLabel("Number of Epochs:"), 1, 0)
        right_layout.addWidget(self.epochs_spinbox, 1, 1)

        right_layout.addWidget(QLabel("Early Stopping Patience:"), 2, 0)
        right_layout.addWidget(self.patience_spinbox, 2, 1)

        right_layout.addWidget(QLabel("Batch Size:"), 3, 0)
        right_layout.addWidget(self.batch_spinbox, 3, 1)

        right_layout.addWidget(self.auto_batch_checkbox, 4, 0, 1, 2)

        right_frame = QFrame()
        right_frame.setLayout(right_layout)

        # Main layout
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(left_frame)
        frame_layout.addWidget(right_frame)

        main_layout = QVBoxLayout()
        main_layout.addLayout(frame_layout)
        main_layout.addWidget(self.train_button)
        main_layout.addWidget(self.log_box)

        self.central.setLayout(main_layout)

    def _create_toolbar(self):
        toolbar = QToolBar("Extra Tools")
        self.addToolBar(toolbar)

        self.open_labelme_btn = QPushButton("Open Labelme")
        self.organize_dataset_btn = QPushButton("Organize Dataset")
        self.export_model_btn = QPushButton("Export Model")

        toolbar.addWidget(self.open_labelme_btn)
        toolbar.addWidget(self.organize_dataset_btn)
        toolbar.addWidget(self.export_model_btn)

    def _connect_signals(self):
        self.dataset_button.clicked.connect(self.open_yaml_dialog)
        self.output_button.clicked.connect(self.open_output_dialog)

        self.auto_batch_checkbox.toggled.connect(
            self.batch_spinbox.setDisabled
        )

        self.train_button.clicked.connect(self.start_training)
        self.open_labelme_btn.clicked.connect(self.launch_labelme)
        self.organize_dataset_btn.clicked.connect(self.open_organizer)
        self.export_model_btn.clicked.connect(self.export_model)
     

    def open_output_dialog(self):
        # Use the static method getOpenFileName
        # Arguments: parent, title, default directory, file filters
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks, # type: ignore
            
        )

        if path:
            self.output_linedit.setText(f"{path}")  
            
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
            self.dataset_linedit.setText(f"{filename}") 
            
    def open_organizer(self):
        organizer = OrganizerWindow(app_state=self.state)
        organizer.exec()
        
    def export_model(self):
        exporter = Exporter(app_state=self.state)
        exporter.exec()
    
    def load_state(self):
        state = self.state

        self.dataset_linedit.setText(
        state.get("trainr.dataset", "")
        )

        self.output_linedit.setText(
        state.get("trainr.output", "")
        )

        self.model_combobox.setCurrentIndex(
        state.get("trainr.model", 1)
        )
        
        self.resolution_spinbox.setValue(
        state.get("trainr.resolution", 640)
        )
        
        self.epochs_spinbox.setValue(
        state.get("trainr.epochs", 100)
        )
        
        self.patience_spinbox.setValue(
        state.get("trainr.patience", 30)
        )
        
        self.batch_spinbox.setValue(
        state.get("trainr.batch_size", 16)
        )
        
        self.batch_spinbox.setEnabled(
        not state.get("trainr.auto_batch", True)
        )
        
        self.auto_batch_checkbox.setChecked(
        state.get("trainr.auto_batch", True)
        )
        
    def bind_state(self):
        s = self.state

        self.dataset_linedit.textChanged.connect(
        lambda v: s.set("trainr.dataset", v)
        )

        self.output_linedit.textChanged.connect(
        lambda v: s.set("trainr.output", v)
        )

        self.model_combobox.currentIndexChanged.connect(
        lambda v: s.set("trainr.model", v)
        )

        self.resolution_spinbox.valueChanged.connect(
        lambda v: s.set("trainr.resolution", v)
        )

        self.epochs_spinbox.valueChanged.connect(
        lambda v: s.set("trainr.epochs", v)
        )

        self.patience_spinbox.valueChanged.connect(
        lambda v: s.set("trainr.patience", v)
        )

        self.batch_spinbox.valueChanged.connect(
        lambda v: s.set("trainr.batch_size", v)
        )
        
        self.auto_batch_checkbox.toggled.connect(
        lambda v: s.set("trainr.auto_batch", v)
        )     
          
    def toggle_batch_size(self, checked):
        if checked:
            self.batch_spinbox.setEnabled(False)
        else:
            self.batch_spinbox.setEnabled(True)
            
    def check_yolo_available(self):
        if not YOLO_EXE.exists():
            return False

        test = QProcess()
        test.start(str(YOLO_EXE), ["--version"])
        test.waitForFinished(3000)
        return test.exitCode() == 0


            
    def start_training(self):
        if hasattr(self, "process") and self.process.state() != QProcess.NotRunning:
            self.log_box.appendPlainText("A process is already running.")
            return
        
        if not self.check_yolo_available():
            self.log_box.appendPlainText(
                "ERROR: YOLO CLI not found. Please run the Heavy Installer first."
            )
            return

        dataset = self.dataset_linedit.text().strip()
        output = self.output_linedit.text().strip()

        if not dataset or not output:
            self.log_box.appendPlainText("ERROR: Dataset or output path missing.")
            return

        model_map = {
            0: "yolov8n.pt",
            1: "yolov8s.pt",
            2: "yolov8m.pt",
            3: "yolov8l.pt",
            4: "yolov8x.pt",
        }

        model = model_map[self.model_combobox.currentIndex()]
        imgsz = self.resolution_spinbox.value()
        epochs = self.epochs_spinbox.value()
        patience = self.patience_spinbox.value()

        if self.auto_batch_checkbox.isChecked():
            batch = "-1"
        else:
            batch = str(self.batch_spinbox.value())

        cmd = [
            str(YOLO_EXE),
            "detect", "train",
            f"data={dataset}",
            f"model={MODELS / model}",
            f"imgsz={imgsz}",
            f"epochs={epochs}",
            f"batch={batch}",
            f"patience={patience}",
            f"project={output}",
            "name=train",
            "exist_ok=True",
        ]



        self.log_box.appendPlainText("Starting YOLO training...\n")
        self.log_box.appendPlainText(" ".join(cmd) + "\n")
        
        self.current_job = "train"
        self._run_process(cmd)

    def _run_process(self, cmd):
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)

        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)

        self.process.start(cmd[0], cmd[1:])

    def _read_process_output(self):
        output = self.process.readAllStandardOutput().data().decode(errors="ignore")
        if output:
            self.log_box.appendPlainText(output.rstrip())
            
    def _process_finished(self, exit_code, exit_status):
        if exit_code != 0:
            self.log_box.appendPlainText(
                f"\nProcess '{self.current_job}' failed."
            )
            self.current_job = None
            return

        if self.current_job == "train":
            self.log_box.appendPlainText("\nTraining finished successfully.")
            self.current_job = "export_onnx"
            self._export_onnx()

        elif self.current_job == "export_onnx":
            self.log_box.appendPlainText("\nONNX export finished successfully.")
            self.current_job = None

    def _export_onnx(self):
        output = self.output_linedit.text().strip()
        weights = f"{output}/train/weights/best.pt"

        cmd = [
            str(YOLO_EXE),
            "export",
            f"model={weights}",
            "format=onnx",
            "opset=12",
            "simplify=True",
        ]

        self.log_box.appendPlainText("\nExporting ONNX...\n")
        self.current_job = "export_onnx"
        self._run_process(cmd)
        
    def open_output_folder(self):
        path = self.state.get("trainr.output", "")
        if path:
            subprocess.Popen(f'explorer "{path}"')

    def launch_labelme(self):
        QProcess.startDetached(
            str(PYTHON),
            ["-m", "labelme"]
        )


            
app = QApplication(sys.argv)
font = QFont("Segoe UI", 11)  # Windows-safe
app.setFont(font)
QApplication.setStyle("Fusion")
state = AppState(str(CONFIG))

window = MainWindow(state)
window.show()

app.exec()