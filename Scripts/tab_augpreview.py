"""
tab_augpreview.py  —  TRAINR
AugPreviewTab: pick an image folder, apply current Train-tab augmentation
parameters, display a 2×3 grid (original + 5 variants).

Fixes vs previous version
--------------------------
- Thread/worker reference cleared after Qt cleanup so isRunning() never
  throws RuntimeError on deleted C++ object.
- "Refresh" re-runs with the same folder, re-reading current params live.
- "Clear" resets the view back to the placeholder.
- Params are read at click time so they always reflect the spinbox values.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from theme import palette

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
N_VARIANTS = 6   # original + 5 augmented


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _build_transform(params: dict):
    try:
        import albumentations as A
        transforms = []

        if params.get("fliplr", 0) > 0:
            transforms.append(A.HorizontalFlip(p=params["fliplr"]))
        if params.get("flipud", 0) > 0:
            transforms.append(A.VerticalFlip(p=params["flipud"]))
        if params.get("degrees", 0) > 0:
            d = params["degrees"]
            transforms.append(A.Rotate(limit=(-d, d), p=0.7,
                                       border_mode=cv2.BORDER_REFLECT_101))
        h = params.get("hsv_h", 0.015)
        s = params.get("hsv_s", 0.7)
        v = params.get("hsv_v", 0.4)
        if h > 0 or s > 0 or v > 0:
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=int(h * 180),
                sat_shift_limit=int(s * 255),
                val_shift_limit=int(v * 255),
                p=0.8,
            ))

        transforms.append(A.NoOp())
        return A.Compose(transforms)

    except ImportError:
        return None


def _opencv_aug(img: np.ndarray, params: dict) -> np.ndarray:
    out = img.copy()
    if random.random() < params.get("fliplr", 0):
        out = cv2.flip(out, 1)
    if random.random() < params.get("flipud", 0):
        out = cv2.flip(out, 0)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + params.get("hsv_h", 0) * 180
                   * random.uniform(-1, 1)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (
        1 + params.get("hsv_s", 0) * random.uniform(-0.5, 0.5)), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (
        1 + params.get("hsv_v", 0) * random.uniform(-0.5, 0.5)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────────

class _PreviewWorker(QObject):
    finished = Signal(object)   # list[np.ndarray] in RGB
    error    = Signal(str)

    def __init__(self, folder: Path, params: dict):
        super().__init__()
        self._folder = folder
        self._params = params

    def run(self):
        try:
            images = [
                p for p in self._folder.rglob("*")
                if p.suffix.lower() in IMAGE_EXTS
            ]
            if not images:
                raise RuntimeError("No images found in the selected folder.")

            src_path = random.choice(images)
            src = cv2.imread(str(src_path))
            if src is None:
                raise RuntimeError(f"Could not read image:\n{src_path}")

            h, w = src.shape[:2]
            if max(h, w) > 640:
                scale = 640 / max(h, w)
                src = cv2.resize(src, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)

            transform = _build_transform(self._params)
            variants: list[np.ndarray] = [
                cv2.cvtColor(src, cv2.COLOR_BGR2RGB)]   # original first

            for _ in range(N_VARIANTS - 1):
                aug = (transform(image=src)["image"]
                       if transform is not None
                       else _opencv_aug(src, self._params))
                variants.append(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))

            self.finished.emit(variants)

        except Exception as exc:
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Canvas
# ──────────────────────────────────────────────────────────────────────────────

class _PreviewCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self._fig = Figure(tight_layout=True)
        self._fig.patch.set_facecolor("none")
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def render(self, variants: list[np.ndarray]):
        self._fig.clear()
        axes = self._fig.subplots(2, 3)
        titles = ["Original"] + [f"Variant {i}" for i in range(1, N_VARIANTS)]
        pal = palette()
        for ax, img, title in zip(axes.flatten(), variants, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=8, color=pal["TEXT_2"])
            ax.axis("off")
        self._fig.set_size_inches(9, 5)
        self.draw()


# ──────────────────────────────────────────────────────────────────────────────
# AugPreviewTab
# ──────────────────────────────────────────────────────────────────────────────

class AugPreviewTab(QWidget):
    def __init__(self, train_tab=None, parent=None):
        super().__init__(parent)
        self._train_tab  = train_tab
        self._thread: QThread | None = None   # cleared after Qt deletes it
        self._worker: _PreviewWorker | None = None
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        # Input frame
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        g = QGridLayout(frame)
        g.setContentsMargins(10, 8, 10, 8)
        g.setSpacing(8)
        g.setColumnStretch(1, 1)

        g.addWidget(QLabel("Image folder:"), 0, 0)
        self._img_folder = QLineEdit()
        self._img_folder.setPlaceholderText("Folder of training images")
        self._img_folder.setReadOnly(True)
        g.addWidget(self._img_folder, 0, 1)
        b1 = QPushButton("Browse")
        b1.setFixedWidth(76)
        b1.clicked.connect(self._browse_img)
        g.addWidget(b1, 0, 2)

        g.addWidget(QLabel("Dataset YAML:"), 1, 0)
        self._yaml = QLineEdit()
        self._yaml.setPlaceholderText("dataset.yaml  (optional)")
        self._yaml.setReadOnly(True)
        g.addWidget(self._yaml, 1, 1)
        b2 = QPushButton("Browse")
        b2.setFixedWidth(76)
        b2.clicked.connect(self._browse_yaml)
        g.addWidget(b2, 1, 2)

        lay.addWidget(frame)

        # Hint
        hint = QLabel(
            "Reads augmentation parameters from the Train tab  ·  "
            "each Refresh picks a new random image with new variants"
        )
        hint.setStyleSheet(
            f"font-size: 8.5pt; color: {palette()['TEXT_3']};")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._gen_btn = QPushButton("Generate Preview")
        self._gen_btn.setObjectName("primaryBtn")
        self._gen_btn.setEnabled(False)
        self._gen_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self._gen_btn)

        self._refresh_btn = QPushButton("↺  Refresh")
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self._refresh_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self._clear_btn)

        btn_row.addStretch()
        lay.addLayout(btn_row)

        # Status
        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("font-size: 9pt; color: #888;")
        lay.addWidget(self._status)

        # Canvas in scroll area
        self._canvas = _PreviewCanvas()
        self._canvas.setVisible(False)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._canvas)
        scroll.setVisible(False)
        self._scroll = scroll
        lay.addWidget(scroll, stretch=1)

        # Placeholder
        self._placeholder = QLabel(
            "Select an image folder and click Generate Preview")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-size: 12pt;")
        self._placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self._placeholder, stretch=1)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _browse_img(self):
        d = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if d:
            self._img_folder.setText(d)
            self._gen_btn.setEnabled(True)

    def _browse_yaml(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select YAML", "",
            "YAML (*.yaml *.yml);;All Files (*.*)")
        if f:
            self._yaml.setText(f)

    def _on_generate(self):
        folder = Path(self._img_folder.text().strip())
        if not folder.is_dir():
            QMessageBox.warning(self, "Invalid folder",
                                "Please select a valid image folder.")
            return

        # Safe running check — don't trust the deleted C++ object
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    return          # already running, ignore click
            except RuntimeError:
                pass                # Qt already deleted the thread — safe to proceed
            self._thread = None
            self._worker = None

        params = self._read_params()

        self._gen_btn.setEnabled(False)
        self._refresh_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)
        self._status.setText("Generating…")
        self._scroll.setVisible(False)
        self._canvas.setVisible(False)
        self._placeholder.setVisible(False)

        self._thread = QThread()
        self._worker = _PreviewWorker(folder, params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        # Clear our references once Qt has cleaned up
        self._thread.finished.connect(self._clear_thread_refs)
        self._thread.start()

    def _on_done(self, variants: list):
        self._gen_btn.setEnabled(True)
        self._refresh_btn.setEnabled(True)
        self._clear_btn.setEnabled(True)
        self._status.setText(
            f"Showing augmented variants  ·  "
            f"folder: {Path(self._img_folder.text()).name}  ·  "
            f"params from Train tab")
        self._canvas.render(variants)
        self._canvas.setVisible(True)
        self._scroll.setVisible(True)

    def _on_error(self, msg: str):
        self._gen_btn.setEnabled(True)
        self._refresh_btn.setEnabled(bool(self._img_folder.text()))
        self._clear_btn.setEnabled(False)
        self._status.setText("")
        self._placeholder.setVisible(True)
        QMessageBox.critical(self, "Preview error", msg)

    def _on_clear(self):
        self._canvas.setVisible(False)
        self._scroll.setVisible(False)
        self._placeholder.setVisible(True)
        self._status.setText("")
        self._refresh_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)

    def _clear_thread_refs(self):
        """Called after Qt deletes the thread — nulls our references safely."""
        self._thread = None
        self._worker = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _read_params(self) -> dict:
        """Read live values from TrainTab at click time."""
        if self._train_tab is None:
            return {}
        t = self._train_tab
        return {
            "fliplr":     t.fliplr_spinbox.value(),
            "flipud":     t.flipud_spinbox.value(),
            "degrees":    t.degrees_spinbox.value(),
            "hsv_h":      t.hsv_h_spinbox.value(),
            "hsv_s":      t.hsv_s_spinbox.value(),
            "hsv_v":      t.hsv_v_spinbox.value(),
            "mosaic":     t.mosaic_spinbox.value(),
            "mixup":      t.mixup_spinbox.value(),
            "copy_paste": t.copy_paste_spinbox.value(),
        }

    def set_train_tab(self, train_tab):
        self._train_tab = train_tab
