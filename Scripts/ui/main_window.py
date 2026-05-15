"""
ui/main_window.py  —  TRAINR
MainWindow: wires sidebar, tab bar, pages, log, status strip, and training process.
Contains no widget construction — all sub-widgets are imported from their own modules.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore    import Qt, QProcess
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
    QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from core.app_state  import AppState
from paths           import YOLO_EXE, LABELME, MODELS
from theme           import apply_theme, auto_titlebar, current_theme

from ui.widgets   import TabBar, hsep
from ui.titlebar  import TitleBar, StatusStrip
from ui.sidebar   import Sidebar
from ui.train_tab import TrainTab

from ui.tabs.tab_curves     import CurvesTab
from ui.tabs.tab_augpreview import AugPreviewTab
from ui.tabs.tab_onnx       import OnnxTab

from ui.dialogs.organizer        import OrganizerWindow
from ui.dialogs.exporter         import Exporter
from ui.dialogs.analyzer_ui      import DatasetVisualizer
from ui.dialogs.emptytxtgenerator import EmptyLabelsDialog
from ui.dialogs.class_renamer    import ClassRenamerDialog
from ui.dialogs.dialogs_extra    import ResumeTrainingDialog, RunComparisonDialog


class MainWindow(QMainWindow):

    _MODEL_MAP = {
        0: "yolov8n.pt",    1: "yolov8s.pt",    2: "yolov8m.pt",
        3: "yolov8l.pt",    4: "yolov8x.pt",
        5: "yolov8n-seg.pt", 6: "yolov8s-seg.pt", 7: "yolov8m-seg.pt",
        8: "yolov8l-seg.pt", 9: "yolov8x-seg.pt",
    }

    def __init__(self, app_state: AppState | None = None):
        super().__init__()
        self.state        = app_state
        self.current_job: str | None = None

        self.setWindowTitle("TRAINR")
        self.resize(1150, 680)
        self.setMinimumSize(860, 580)

        self._build_ui()
        self._connect_signals()

        if self.state:
            self.load_state()
            self.bind_state()

    # ──────────────────────────────────────────────────────────────────────────
    # UI assembly
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Title bar
        self.titlebar = TitleBar()
        root.addWidget(self.titlebar)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)

        self.sidebar = Sidebar()
        splitter.addWidget(self.sidebar)

        # Main column
        main_col     = QWidget()
        main_col_lay = QVBoxLayout(main_col)
        main_col_lay.setContentsMargins(0, 0, 0, 0)
        main_col_lay.setSpacing(0)

        self.tab_bar = TabBar(["Train", "Curves", "Aug. Preview", "ONNX / HEF"])
        main_col_lay.addWidget(self.tab_bar)

        tb_sep = QFrame()
        tb_sep.setFrameShape(QFrame.Shape.HLine)
        tb_sep.setFixedHeight(1)
        main_col_lay.addWidget(tb_sep)

        self.pages          = QStackedWidget()
        self.train_tab      = TrainTab()
        self.curves_tab     = CurvesTab()
        self.aug_preview_tab= AugPreviewTab(train_tab=self.train_tab)
        self.onnx_tab       = OnnxTab(app_state=self.state)

        for tab in [self.train_tab, self.curves_tab,
                    self.aug_preview_tab, self.onnx_tab]:
            self.pages.addWidget(tab)
        main_col_lay.addWidget(self.pages, stretch=1)

        # Log panel
        log_panel     = QWidget()
        log_panel_lay = QVBoxLayout(log_panel)
        log_panel_lay.setContentsMargins(0, 0, 0, 0)
        log_panel_lay.setSpacing(0)

        log_top     = QWidget()
        log_top.setFixedHeight(28)
        log_top_lay = QHBoxLayout(log_top)
        log_top_lay.setContentsMargins(10, 0, 10, 0)
        log_top_lay.setSpacing(0)
        log_lbl = QLabel("Log")
        log_lbl.setStyleSheet(
            "font-size: 9pt; font-weight: 600; color: #9E9C97; background: transparent;")
        log_top_lay.addWidget(log_lbl)
        log_top_lay.addStretch()
        log_panel_lay.addWidget(log_top)
        log_panel_lay.addWidget(_hline())

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(5000)
        self.log_box.setFixedHeight(130)
        self.log_box.setPlaceholderText("Training logs will appear here…")
        log_panel_lay.addWidget(self.log_box)
        main_col_lay.addWidget(log_panel)

        splitter.addWidget(main_col)
        splitter.setSizes([240, 910])
        root.addWidget(splitter, stretch=1)

        self.status_strip = StatusStrip()
        root.addWidget(self.status_strip)

    # ──────────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.tab_bar.tab_changed.connect(self.pages.setCurrentIndex)

        # Sidebar
        self.sidebar.dataset_button.clicked.connect(self._browse_yaml)
        self.sidebar.output_button.clicked.connect(self._browse_output)
        self.sidebar.analyze_btn.clicked.connect(
            lambda: DatasetVisualizer().exec())
        self.sidebar.class_rename_btn.clicked.connect(
            lambda: ClassRenamerDialog(app_state=self.state, parent=self).exec())
        self.sidebar.organize_btn.clicked.connect(
            lambda: OrganizerWindow(app_state=self.state).exec())
        self.sidebar.emptylabels_btn.clicked.connect(
            lambda: EmptyLabelsDialog().exec())

        # Title bar
        self.titlebar.labelme_clicked.connect(
            lambda: QProcess.startDetached(str(LABELME)))
        self.titlebar.language_clicked.connect(self._open_language)
        self.titlebar.reset_clicked.connect(self._reset_params)
        self.titlebar.resume_clicked.connect(
            lambda: ResumeTrainingDialog(app_state=self.state, parent=self).exec())
        self.titlebar.compare_clicked.connect(
            lambda: RunComparisonDialog(app_state=self.state, parent=self).exec())
        self.titlebar.theme_toggled.connect(self._toggle_theme)

        # Status strip
        self.status_strip.start_clicked.connect(self.start_training)

        # Cross-tab
        self.curves_tab.last_run_ready.connect(self.train_tab.update_last_run)

    # ──────────────────────────────────────────────────────────────────────────
    # Theme
    # ──────────────────────────────────────────────────────────────────────────

    def _toggle_theme(self):
        new = "light" if current_theme() == "dark" else "dark"
        apply_theme(QApplication.instance(), new)
        auto_titlebar(self)
        self.titlebar.set_theme_label(new)
        self.tab_bar.refresh_styles()
        self.status_strip.refresh_color()
        if self.state:
            self.state.set("ui.theme", new)

    # ──────────────────────────────────────────────────────────────────────────
    # File dialogs
    # ──────────────────────────────────────────────────────────────────────────

    def _browse_yaml(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select YAML", "",
            "YAML Files (*.yaml *.yml);;All Files (*.*)")
        if f:
            self.sidebar.dataset_linedit.setText(f)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", "",
            QFileDialog.Option.ShowDirsOnly)
        if d:
            self.sidebar.output_linedit.setText(d)

    # ──────────────────────────────────────────────────────────────────────────
    # Misc actions
    # ──────────────────────────────────────────────────────────────────────────

    def _open_language(self):
        QMessageBox.information(self, "Language",
            "Additional languages coming soon.\nCurrently: English")

    def _reset_params(self):
        t = self.train_tab
        t.dropout_spinbox.setValue(0.0)
        t.weight_decay_spinbox.setValue(0.0005)
        t.label_smoothing_spinbox.setValue(0.0)
        t.warmup_epochs_spinbox.setValue(3.0)
        t.cos_lr_checkbox.setChecked(False)
        t.mosaic_spinbox.setValue(1.0)
        t.mixup_spinbox.setValue(0.0)
        t.copy_paste_spinbox.setValue(0.0)
        t.degrees_spinbox.setValue(0.0)
        t.fliplr_spinbox.setValue(0.5)
        t.flipud_spinbox.setValue(0.0)
        t.hsv_h_spinbox.setValue(0.015)
        t.hsv_s_spinbox.setValue(0.7)
        t.hsv_v_spinbox.setValue(0.4)
        self.log_box.appendPlainText(
            "Regularization and augmentation parameters reset to defaults.")

    # ──────────────────────────────────────────────────────────────────────────
    # State persistence
    # ──────────────────────────────────────────────────────────────────────────

    def load_state(self):
        s = self.state
        t = self.train_tab
        o = self.onnx_tab

        saved_theme = s.get("ui.theme", "dark")
        apply_theme(QApplication.instance(), saved_theme)
        auto_titlebar(self)
        self.titlebar.set_theme_label(saved_theme)
        self.tab_bar.refresh_styles()
        self.status_strip.refresh_color()

        saved_tab = s.get("ui.active_tab", 0)
        self.pages.setCurrentIndex(saved_tab)
        self.tab_bar._select(saved_tab)

        self.sidebar.dataset_linedit.setText(s.get("trainr.dataset", ""))
        self.sidebar.output_linedit.setText(s.get("trainr.output", ""))
        self.sidebar.set_model_index(s.get("trainr.model", 1))

        t.resolution_spinbox.setValue(s.get("trainr.resolution", 640))
        t.epochs_spinbox.setValue(s.get("trainr.epochs", 100))
        t.patience_spinbox.setValue(s.get("trainr.patience", 30))
        t.batch_spinbox.setValue(s.get("trainr.batch_size", 16))
        t.workers_spinbox.setValue(s.get("trainr.workers", 8))
        t.batch_spinbox.setEnabled(not s.get("trainr.auto_batch", True))
        t.auto_batch_checkbox.setChecked(s.get("trainr.auto_batch", True))

        t.dropout_spinbox.setValue(s.get("trainr.dropout", 0.0))
        t.weight_decay_spinbox.setValue(s.get("trainr.weight_decay", 0.0005))
        t.label_smoothing_spinbox.setValue(s.get("trainr.label_smoothing", 0.0))
        t.warmup_epochs_spinbox.setValue(s.get("trainr.warmup_epochs", 3.0))
        t.cos_lr_checkbox.setChecked(s.get("trainr.cos_lr", False))

        t.mosaic_spinbox.setValue(s.get("trainr.mosaic", 1.0))
        t.mixup_spinbox.setValue(s.get("trainr.mixup", 0.0))
        t.copy_paste_spinbox.setValue(s.get("trainr.copy_paste", 0.0))
        t.degrees_spinbox.setValue(s.get("trainr.degrees", 0.0))
        t.fliplr_spinbox.setValue(s.get("trainr.fliplr", 0.5))
        t.flipud_spinbox.setValue(s.get("trainr.flipud", 0.0))
        t.hsv_h_spinbox.setValue(s.get("trainr.hsv_h", 0.015))
        t.hsv_s_spinbox.setValue(s.get("trainr.hsv_s", 0.7))
        t.hsv_v_spinbox.setValue(s.get("trainr.hsv_v", 0.4))

        o.onnx_input.setText(s.get("onnx.onnx_path", ""))
        o.yaml_input.setText(s.get("onnx.yaml_path", ""))
        o.out_input.setText(s.get("onnx.output_folder", ""))
        o.resolution_input.setValue(s.get("onnx.resolution", 640))
        o.model_name_input.setText(s.get("onnx.model_name", ""))

    def bind_state(self):
        s  = self.state
        t  = self.train_tab
        o  = self.onnx_tab
        sb = self.sidebar

        sb.dataset_linedit.textChanged.connect(lambda v: s.set("trainr.dataset", v))
        sb.output_linedit.textChanged.connect(lambda v: s.set("trainr.output", v))
        sb.model_changed.connect(lambda v: s.set("trainr.model", v))
        self.tab_bar.tab_changed.connect(lambda v: s.set("ui.active_tab", v))

        t.resolution_spinbox.valueChanged.connect(lambda v: s.set("trainr.resolution", v))
        t.epochs_spinbox.valueChanged.connect(lambda v: s.set("trainr.epochs", v))
        t.patience_spinbox.valueChanged.connect(lambda v: s.set("trainr.patience", v))
        t.batch_spinbox.valueChanged.connect(lambda v: s.set("trainr.batch_size", v))
        t.auto_batch_checkbox.toggled.connect(lambda v: s.set("trainr.auto_batch", v))
        t.workers_spinbox.valueChanged.connect(lambda v: s.set("trainr.workers", v))
        t.dropout_spinbox.valueChanged.connect(lambda v: s.set("trainr.dropout", v))
        t.weight_decay_spinbox.valueChanged.connect(lambda v: s.set("trainr.weight_decay", v))
        t.label_smoothing_spinbox.valueChanged.connect(lambda v: s.set("trainr.label_smoothing", v))
        t.warmup_epochs_spinbox.valueChanged.connect(lambda v: s.set("trainr.warmup_epochs", v))
        t.cos_lr_checkbox.toggled.connect(lambda v: s.set("trainr.cos_lr", v))
        t.mosaic_spinbox.valueChanged.connect(lambda v: s.set("trainr.mosaic", v))
        t.mixup_spinbox.valueChanged.connect(lambda v: s.set("trainr.mixup", v))
        t.copy_paste_spinbox.valueChanged.connect(lambda v: s.set("trainr.copy_paste", v))
        t.degrees_spinbox.valueChanged.connect(lambda v: s.set("trainr.degrees", v))
        t.fliplr_spinbox.valueChanged.connect(lambda v: s.set("trainr.fliplr", v))
        t.flipud_spinbox.valueChanged.connect(lambda v: s.set("trainr.flipud", v))
        t.hsv_h_spinbox.valueChanged.connect(lambda v: s.set("trainr.hsv_h", v))
        t.hsv_s_spinbox.valueChanged.connect(lambda v: s.set("trainr.hsv_s", v))
        t.hsv_v_spinbox.valueChanged.connect(lambda v: s.set("trainr.hsv_v", v))

        o.onnx_input.textChanged.connect(lambda v: s.set("onnx.onnx_path", v))
        o.yaml_input.textChanged.connect(lambda v: s.set("onnx.yaml_path", v))
        o.out_input.textChanged.connect(lambda v: s.set("onnx.output_folder", v))
        o.resolution_input.valueChanged.connect(lambda v: s.set("onnx.resolution", v))
        o.model_name_input.textChanged.connect(lambda v: s.set("onnx.model_name", v))

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def check_yolo_available(self) -> bool:
        if not YOLO_EXE.exists():
            return False
        test = QProcess()
        test.start(str(YOLO_EXE), ["--version"])
        test.waitForFinished(3000)
        return test.exitCode() == 0

    def start_training(self):
        if hasattr(self, "process") and \
                self.process.state() != QProcess.ProcessState.NotRunning:
            self.log_box.appendPlainText("A process is already running.")
            return

        if not self.check_yolo_available():
            self.log_box.appendPlainText(
                "ERROR: YOLO CLI not found. Run the Heavy Installer.")
            return

        dataset = self.sidebar.dataset_linedit.text().strip()
        output  = self.sidebar.output_linedit.text().strip()
        if not dataset or not output:
            self.log_box.appendPlainText("ERROR: Dataset or output path missing.")
            return

        model_idx  = self.sidebar.current_model_index()
        model_name = self._MODEL_MAP.get(model_idx, "yolov8s.pt")
        task       = "segment" if "-seg" in model_name else "detect"
        t          = self.train_tab
        batch      = "-1" if t.auto_batch_checkbox.isChecked() \
                     else str(t.batch_spinbox.value())

        cmd = [
            str(YOLO_EXE), task, "train",
            f"data={dataset}",
            f"model={MODELS / model_name}",
            f"imgsz={t.resolution_spinbox.value()}",
            f"epochs={t.epochs_spinbox.value()}",
            f"batch={batch}",
            f"patience={t.patience_spinbox.value()}",
            f"workers={t.workers_spinbox.value()}",
            f"project={output}", "name=train", "exist_ok=True",
            f"dropout={t.dropout_spinbox.value()}",
            f"weight_decay={t.weight_decay_spinbox.value()}",
            f"label_smoothing={t.label_smoothing_spinbox.value()}",
            f"warmup_epochs={t.warmup_epochs_spinbox.value()}",
            f"cos_lr={t.cos_lr_checkbox.isChecked()}",
            f"mosaic={t.mosaic_spinbox.value()}",
            f"mixup={t.mixup_spinbox.value()}",
            f"copy_paste={t.copy_paste_spinbox.value()}",
            f"degrees={t.degrees_spinbox.value()}",
            f"fliplr={t.fliplr_spinbox.value()}",
            f"flipud={t.flipud_spinbox.value()}",
            f"hsv_h={t.hsv_h_spinbox.value()}",
            f"hsv_s={t.hsv_s_spinbox.value()}",
            f"hsv_v={t.hsv_v_spinbox.value()}",
        ]

        self.log_box.appendPlainText(f"\nStarting YOLO {task} training…\n")
        self.log_box.appendPlainText(" ".join(cmd) + "\n")
        self.status_strip.set_training(model_name)
        self.current_job = "train"
        self._run_process(cmd)

    def _run_process(self, cmd: list[str]):
        self.process = QProcess(self)
        self.process.errorOccurred.connect(
            lambda e: self.log_box.appendPlainText(f"Process error: {e}"))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._process_finished)
        self.process.start(cmd[0], cmd[1:])

    def _read_output(self):
        raw = self.process.readAllStandardOutput().data().decode(errors="ignore")
        if not raw:
            return
        self.log_box.appendPlainText(raw.rstrip())
        for line in raw.splitlines():
            if "/" not in line:
                continue
            for token in line.split():
                if "/" in token:
                    try:
                        c, tot = token.split("/", 1)
                        self.status_strip.set_epoch(int(c), int(tot))
                        return
                    except ValueError:
                        pass

    def _process_finished(self, exit_code, _exit_status):
        if exit_code != 0:
            self.log_box.appendPlainText(f"\nProcess '{self.current_job}' failed.")
            self.status_strip.set_failed(self.current_job or "")
            self.current_job = None
            return

        if self.current_job == "train":
            self.log_box.appendPlainText("\nTraining finished successfully.")
            self.status_strip.set_exporting()
            self.current_job = "export_onnx"
            self._export_onnx()
        elif self.current_job == "export_onnx":
            self.log_box.appendPlainText("\nONNX export finished.")
            self.status_strip.set_done()
            self.current_job = None
            csv = (Path(self.sidebar.output_linedit.text())
                   / "train" / "results.csv")
            if csv.exists():
                self.curves_tab.load_csv(str(csv))
                self.tab_bar._select(1)

    def _export_onnx(self):
        output  = self.sidebar.output_linedit.text().strip()
        weights = f"{output}/train/weights/best.pt"
        cmd = [str(YOLO_EXE), "export", f"model={weights}",
               "format=onnx", "opset=11", "simplify=True"]
        self.log_box.appendPlainText("\nExporting ONNX…\n")
        self._run_process(cmd)


# ── private helpers ───────────────────────────────────────────────────────────

def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    return f
