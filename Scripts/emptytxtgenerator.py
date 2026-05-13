"""
empty_labels.py
---------------
Generates empty .txt label files alongside images that have no label yet.
Used to mark negative / background images in a YOLO dataset.

Supports .jpg .jpeg .png .bmp .tiff
Recursively scans subfolders if requested.

Standalone:  python empty_labels.py
From main:   EmptyLabelsDialog(parent=self).exec()
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ──────────────────────────────────────────────────────────────────────────────
# Backend
# ──────────────────────────────────────────────────────────────────────────────

def generate_empty_labels(
    folder: Path | str,
    recursive: bool = True,
) -> dict:
    """
    For every image in `folder` that has no matching .txt file, create an
    empty .txt file in the same directory as the image.

    Returns
    -------
    dict with keys: created (int), skipped (int), total_images (int)
    """
    folder = Path(folder)
    pattern = "**/*" if recursive else "*"

    created = 0
    skipped = 0
    total   = 0

    for img_path in sorted(folder.glob(pattern)):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        total += 1
        label = img_path.with_suffix(".txt")

        if label.exists():
            skipped += 1
        else:
            label.touch()
            created += 1

    return {"created": created, "skipped": skipped, "total_images": total}


# ──────────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────────

class _Worker(QObject):
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, folder: Path, recursive: bool):
        super().__init__()
        self._folder    = folder
        self._recursive = recursive

    def run(self):
        try:
            result = generate_empty_labels(self._folder, self._recursive)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────

class EmptyLabelsDialog(QDialog):
    def __init__(self, app_state=None, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.setWindowTitle("Generate Empty Labels")
        self.setMinimumWidth(480)
        self.resize(50, 130)
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(14, 14, 14, 14)

        # ── input frame ──────────────────────────────────────────────────────
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        grid = QGridLayout(frame)
        grid.setSpacing(8)
        grid.setContentsMargins(10, 10, 10, 10)

        grid.addWidget(QLabel("Images folder:"), 0, 0)

        self._path_input = QLineEdit()
        self._path_input.setPlaceholderText("Folder containing your negative images")
        self._path_input.setReadOnly(True)
        grid.addWidget(self._path_input, 0, 1)

        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        grid.addWidget(browse_btn, 0, 2)


        grid.setColumnStretch(1, 1)
        main_layout.addWidget(frame)

        # ── action button ─────────────────────────────────────────────────────
        self._run_btn = QPushButton("Generate Empty Labels")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run_clicked)
        main_layout.addWidget(self._run_btn)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Images Folder", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,  # type: ignore
        )
        if folder:
            self._path_input.setText(folder)
            self._run_btn.setEnabled(True)

    def _on_run_clicked(self):
        folder = Path(self._path_input.text().strip())
        if not folder.is_dir():
            QMessageBox.warning(self, "Invalid path", "Please select a valid folder.")
            return

        self._run_btn.setEnabled(False)

        recursive = True

        self._thread = QThread()
        self._worker = _Worker(folder, recursive)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_finished(self, result: dict):
        self._run_btn.setEnabled(True)
        QMessageBox.information(
            self,
            "Done",
            f"Finished scanning {result['total_images']} image(s).\n\n"
            f"Empty labels created:  {result['created']}\n"
            f"Already had a label:   {result['skipped']}",
        )

    def _on_error(self, msg: str):
        self._run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 11)
    app.setFont(font)
    QApplication.setStyle("Fusion")
    window = EmptyLabelsDialog()
    window.show()
    sys.exit(app.exec())