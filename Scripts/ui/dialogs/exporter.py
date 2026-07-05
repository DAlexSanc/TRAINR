"""
ui/dialogs/exporter.py  —  TRAINR
Exporter dialog — thin UI shell over ExportWorker (core/export_worker.py).
"""
from __future__ import annotations

import sys

from PySide6.QtCore    import QSize, QThread
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame,
    QGridLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout,
)

from core.export_worker import ExportWorker
from core.app_state     import AppState
from theme import auto_titlebar, apply_theme


class Exporter(QDialog):
    def __init__(self, app_state: AppState | None = None):
        super().__init__()
        self.app_state = app_state
        self.setWindowTitle("Model Exporter")
        self.setMinimumSize(QSize(500, 300))
        self._build()
        auto_titlebar(self)

    def _build(self):
        main_layout = QVBoxLayout(self)

        # ── Paths frame ───────────────────────────────────────────────────────
        frame_top    = QFrame()
        frame_top.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout_top   = QGridLayout(frame_top)

        self.onnx_path_input  = QLineEdit()
        self.output_input     = QLineEdit()
        self.yaml_file_input  = QLineEdit()

        for row, (label_text, widget, browse_slot) in enumerate([
            ("ONNX File Path:",    self.onnx_path_input,  self._browse_onnx),
            ("Output Folder Path:",self.output_input,     self._browse_output),
            ("YAML File Path:",    self.yaml_file_input,  self._browse_yaml),
        ]):
            btn = QPushButton("Browse")
            btn.clicked.connect(browse_slot)
            layout_top.addWidget(QLabel(label_text), row, 0)
            layout_top.addWidget(widget,              row, 1, 1, 3)
            layout_top.addWidget(btn,                 row, 4)

        # ── Params frame ──────────────────────────────────────────────────────
        frame_bot  = QFrame()
        frame_bot.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout_bot = QGridLayout(frame_bot)

        self.resolution_input = QSpinBox()
        self.resolution_input.setRange(160, 2048)
        self.resolution_input.setValue(640)
        self.model_name_input = QLineEdit()

        layout_bot.addWidget(QLabel("Model Resolution:"), 0, 0)
        layout_bot.addWidget(self.resolution_input,       0, 1)
        layout_bot.addWidget(QLabel("Model Name:"),       1, 0)
        layout_bot.addWidget(self.model_name_input,       1, 1)

        self.export_button = QPushButton("Export Model")
        self.export_button.clicked.connect(self._start_export)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(5000)
        self.log_box.setPlaceholderText("Logs will appear here…")

        main_layout.addWidget(frame_top)
        main_layout.addWidget(frame_bot)
        main_layout.addWidget(self.export_button)
        main_layout.addWidget(self.log_box)

    # ── Browse slots ──────────────────────────────────────────────────────────

    def _browse_onnx(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX File", "",
            "ONNX Files (*.onnx);;All Files (*.*)")
        if f:
            self.onnx_path_input.setText(f)

    def _browse_output(self):
        p = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)  # type: ignore
        if p:
            self.output_input.setText(p)

    def _browse_yaml(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select YAML File", "",
            "YAML Files (*.yaml *.yml);;All Files (*.*)")
        if f:
            self.yaml_file_input.setText(f)

    # ── Export ────────────────────────────────────────────────────────────────

    def _start_export(self):
        self.export_button.setEnabled(False)
        self._thread = QThread()
        self._worker = ExportWorker(
            self.onnx_path_input.text(),
            self.yaml_file_input.text(),
            self.output_input.text(),
            self.resolution_input.value(),
            self.model_name_input.text(),
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
        self.export_button.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_theme(app, "dark")
    w = Exporter()
    w.show()
    from theme import dark_titlebar
    dark_titlebar(w)
    sys.exit(app.exec())
