"""
tab_onnx.py  —  TRAINR
OnnxTab: inline ONNX → HEF export tab.

Runs ExportWorker directly (same backend as the standalone Exporter dialog)
and streams logs into the tab's own log box — no dialog indirection.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from exporter import ExportWorker


class OnnxTab(QWidget):
    """
    Drop-in replacement for the placeholder OnnxTab.
    Runs the full ONNX → HEF → ZIP pipeline inline.
    """

    def __init__(self, app_state=None, parent=None):
        super().__init__(parent)
        self._state  = app_state
        self._thread: QThread | None = None
        self._worker: ExportWorker | None = None
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        # ── Source frame ──────────────────────────────────────────────────
        src = QFrame()
        src.setFrameShape(QFrame.Shape.StyledPanel)
        sg = QGridLayout(src)
        sg.setContentsMargins(10, 10, 10, 10)
        sg.setSpacing(8)
        sg.setColumnStretch(1, 1)

        src_title = QLabel("Source")
        src_title.setStyleSheet("font-size: 9.5pt; font-weight: 600;")
        sg.addWidget(src_title, 0, 0, 1, 3)

        def _path_row(grid, row, label, attr, placeholder, browse_fn):
            grid.addWidget(QLabel(label), row, 0)
            le = QLineEdit()
            le.setPlaceholderText(placeholder)
            le.setReadOnly(True)
            setattr(self, attr, le)
            grid.addWidget(le, row, 1)
            b = QPushButton("…")
            b.setObjectName("iconBtn")
            b.setFixedSize(26, 26)
            b.clicked.connect(browse_fn)
            grid.addWidget(b, row, 2)

        _path_row(sg, 1, "ONNX file:", "onnx_input",
                  "train/weights/best.onnx",
                  lambda: self._browse_file("onnx_input",
                                            "ONNX Files (*.onnx)"))
        _path_row(sg, 2, "Dataset YAML:", "yaml_input",
                  "dataset.yaml",
                  lambda: self._browse_file("yaml_input",
                                            "YAML Files (*.yaml *.yml)"))
        _path_row(sg, 3, "Output folder:", "out_input",
                  "export output folder",
                  lambda: self._browse_dir("out_input"))

        lay.addWidget(src)

        # ── Parameters frame ──────────────────────────────────────────────
        prm = QFrame()
        prm.setFrameShape(QFrame.Shape.StyledPanel)
        pg = QGridLayout(prm)
        pg.setContentsMargins(10, 10, 10, 10)
        pg.setSpacing(8)
        pg.setColumnStretch(1, 1)

        prm_title = QLabel("Parameters")
        prm_title.setStyleSheet("font-size: 9.5pt; font-weight: 600;")
        pg.addWidget(prm_title, 0, 0, 1, 2)

        pg.addWidget(QLabel("Resolution:"), 1, 0)
        self.resolution_input = QSpinBox()
        self.resolution_input.setRange(160, 2048)
        self.resolution_input.setValue(640)
        pg.addWidget(self.resolution_input, 1, 1)

        pg.addWidget(QLabel("Model name:"), 2, 0)
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("e.g. SWT_Benchmark_v3")
        pg.addWidget(self.model_name_input, 2, 1)

        lay.addWidget(prm)

        # ── Export button ─────────────────────────────────────────────────
        self._export_btn = QPushButton("Export Model")
        self._export_btn.setObjectName("primaryBtn")
        self._export_btn.clicked.connect(self._on_export)
        lay.addWidget(self._export_btn)

        # ── Log box ───────────────────────────────────────────────────────
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(5000)
        self.log_box.setPlaceholderText("Export logs will appear here…")
        lay.addWidget(self.log_box, stretch=1)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _browse_file(self, attr: str, filt: str):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select file", "", f"{filt};;All Files (*.*)")
        if f:
            getattr(self, attr).setText(f)

    def _browse_dir(self, attr: str):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            getattr(self, attr).setText(d)

    def _on_export(self):
        onnx = self.onnx_input.text().strip()
        yaml = self.yaml_input.text().strip()
        out  = self.out_input.text().strip()

        if not onnx:
            QMessageBox.warning(self, "Missing input",
                                "Please select an ONNX file.")
            return
        if not yaml:
            QMessageBox.warning(self, "Missing input",
                                "Please select a dataset YAML.")
            return
        if not out:
            QMessageBox.warning(self, "Missing input",
                                "Please select an output folder.")
            return

        if self._thread and self._thread.isRunning():
            self.log_box.appendPlainText("Export already running.")
            return

        self._export_btn.setEnabled(False)
        self.log_box.clear()
        self.log_box.appendPlainText("Starting export…\n")

        self._thread = QThread()
        self._worker = ExportWorker(
            onnx_path   = onnx,
            yaml_path   = yaml,
            output_path = out,
            resolution  = self.resolution_input.value(),
            model_name  = self.model_name_input.text().strip(),
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log_box.appendPlainText)
        self._worker.finished.connect(self._on_done)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_done(self, success: bool, message: str):
        self._export_btn.setEnabled(True)
        self.log_box.appendPlainText(f"\n{'✓ ' if success else '✗ '}{message}")
        if success:
            QMessageBox.information(self, "Export complete", message)
        else:
            QMessageBox.critical(self, "Export failed", message)
