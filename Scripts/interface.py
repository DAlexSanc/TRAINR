"""
interface.py  —  TRAINR
Main window implementing the dashboard layout from the v3 mockup:
  • Left sidebar  : config fields + scrollable model radio list + dataset-tool links
  • Right main col: custom tab bar → stacked param cards / curves / health / ONNX
  • Bottom log    : full-width QPlainTextEdit, always visible
  • Status bar    : slim accent strip — status text left, START TRAINING right
  • Title bar     : icon-only utility buttons (LabelMe, Organise, …)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import (
    Qt, QProcess, QSize, QRect, QPoint,
    Signal,
)
from PySide6.QtGui import (
    QColor, QFont, QPainter, QPen, QBrush,
    QPalette,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QStackedWidget, QStatusBar,
    QVBoxLayout, QWidget, QDialog, QGridLayout,
)

from app_state import AppState
from organizer import OrganizerWindow
from exporter import Exporter
from analyzer_ui import DatasetVisualizer
from emptytxtgenerator import EmptyLabelsDialog
from paths import YOLO_EXE, LABELME, MODELS, CONFIG
from theme import apply_theme, auto_titlebar, current_theme, palette


# ──────────────────────────────────────────────────────────────────────────────
# Helpers / small custom widgets
# ──────────────────────────────────────────────────────────────────────────────

def _icon_btn(icon_char: str, tooltip: str) -> QPushButton:
    b = QPushButton(icon_char)
    b.setObjectName("iconBtn")
    b.setToolTip(tooltip)
    b.setFixedSize(28, 28)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def _link_btn(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("linkBtn")
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    return b


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        "font-size: 8pt; font-weight: 700; letter-spacing: 0.09em;"
        "color: #9E9C97;"
    )
    return lbl


def _hsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    return f


def _vsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setFixedWidth(1)
    return f


class ParamRow(QWidget):
    def __init__(self, label: str, spinbox: QWidget,
                 show_track: bool = False,
                 min_val: float = 0.0, max_val: float = 1.0,
                 parent=None):
        super().__init__(parent)
        self._spinbox    = spinbox
        self._show_track = show_track
        self._min        = min_val
        self._max        = max_val

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        lbl = QLabel(label)
        lbl.setStyleSheet("font-size: 9.5pt; color: #706B63;")
        lbl.setMinimumWidth(90)
        layout.addWidget(lbl)

        if show_track:
            self._track = _TrackBar(spinbox, min_val, max_val)
            self._track.setFixedHeight(14)
            layout.addWidget(self._track, stretch=1)

        layout.addWidget(spinbox)
        spinbox.setFixedWidth(72)

    def value(self):
        return self._spinbox.value()


class _TrackBar(QWidget):
    def __init__(self, spinbox: QWidget, min_val: float, max_val: float):
        super().__init__()
        self._spin = spinbox
        self._min  = min_val
        self._max  = max_val
        self.setMinimumWidth(40)
        if hasattr(spinbox, "valueChanged"):
            spinbox.valueChanged.connect(lambda _: self.update())

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cy = h // 2
        groove_h = 3
        p.setPen(Qt.PenStyle.NoPen)
        pal = palette()
        p.setBrush(QColor(pal["BORDER_S"]))
        p.drawRoundedRect(0, cy - groove_h // 2, w, groove_h, 1, 1)
        rng   = self._max - self._min or 1
        ratio = max(0.0, min(1.0, (self._spin.value() - self._min) / rng))
        fill_w = int(w * ratio)
        if fill_w > 0:
            p.setBrush(QColor(pal["ACCENT"]))
            p.drawRoundedRect(0, cy - groove_h // 2, fill_w, groove_h, 1, 1)
        p.end()


class ParamCard(QFrame):
    def __init__(self, title: str, accent_color: str | None = None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("paramCard")
        self._rows: list[QWidget] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        hdr = QWidget()
        hdr.setFixedHeight(30)
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(10, 0, 10, 0)
        hdr_lay.setSpacing(6)

        dot = QLabel("●")
        dot.setStyleSheet(
            f"font-size: 7pt; color: {accent_color or '#C95F1A'};"
            "background: transparent;"
        )
        hdr_lay.addWidget(dot)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            "font-size: 9.5pt; font-weight: 600; background: transparent;"
        )
        hdr_lay.addWidget(title_lbl)
        hdr_lay.addStretch()

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)

        self._body = QWidget()
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(10, 8, 10, 10)
        self._body_lay.setSpacing(5)

        root.addWidget(hdr)
        root.addWidget(sep)
        root.addWidget(self._body)

    def add_row(self, widget: QWidget):
        self._body_lay.addWidget(widget)

    def add_checkbox(self, cb: QCheckBox):
        self._body_lay.addWidget(cb)


# ──────────────────────────────────────────────────────────────────────────────
# Custom tab bar
# ──────────────────────────────────────────────────────────────────────────────

class TabBar(QWidget):
    tab_changed = Signal(int)

    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
        self._labels  = labels
        self._current = 0
        self.setFixedHeight(36)

        self._btns: list[QPushButton] = []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(0)

        for i, lbl in enumerate(labels):
            btn = QPushButton(lbl)
            btn.setCheckable(True)
            btn.setObjectName("tabBtn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, idx=i: self._select(idx))
            self._btns.append(btn)
            lay.addWidget(btn)

        lay.addStretch()
        self._apply_styles()
        self._btns[0].setChecked(True)

    def _apply_styles(self):
        pal = palette()
        for b in self._btns:
            b.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    border: none;
                    border-bottom: 2px solid transparent;
                    border-radius: 0;
                    padding: 6px 14px;
                    font-size: 10pt;
                    color: {pal['TEXT_3']};
                    min-height: 0;
                }}
                QPushButton:checked {{
                    color: {pal['TEXT']};
                    border-bottom: 2px solid {pal['ACCENT']};
                    font-weight: 600;
                }}
                QPushButton:hover:!checked {{
                    color: {pal['TEXT_2']};
                }}
            """)

    def _select(self, idx: int):
        self._btns[self._current].setChecked(False)
        self._current = idx
        self._btns[idx].setChecked(True)
        self.tab_changed.emit(idx)

    def refresh_styles(self):
        self._apply_styles()

    def current(self) -> int:
        return self._current


# ──────────────────────────────────────────────────────────────────────────────
# Placeholder dialogs
# ──────────────────────────────────────────────────────────────────────────────

class ResumeTrainingDialog(QDialog):
    def __init__(self, app_state=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resume Training")
        self.setMinimumWidth(520)
        self.resize(560, 220)
        self._build()
        auto_titlebar(self)

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        g = QGridLayout(frame)
        g.setSpacing(8)
        g.setContentsMargins(10, 10, 10, 10)
        g.setColumnStretch(1, 1)

        def _row(r, lbl, attr, ph, is_dir=False):
            g.addWidget(QLabel(lbl), r, 0)
            le = QLineEdit()
            le.setPlaceholderText(ph)
            le.setReadOnly(True)
            setattr(self, attr, le)
            g.addWidget(le, r, 1)
            b = QPushButton("Browse")
            b.setFixedWidth(76)
            if is_dir:
                b.clicked.connect(lambda _, a=attr: self._browse_dir(a))
            else:
                b.clicked.connect(lambda _, a=attr, x=ph: self._browse_file(a, x))
            g.addWidget(b, r, 2)

        _row(0, "Checkpoint (.pt):", "_ckpt", "Path to best.pt or last.pt")
        _row(1, "Dataset YAML:", "_yaml", "Path to dataset.yaml")
        _row(2, "Output folder:", "_out", "Output directory", is_dir=True)

        g.addWidget(QLabel("Additional epochs:"), 3, 0)
        self._ep = QSpinBox()
        self._ep.setRange(1, 1000)
        self._ep.setValue(50)
        g.addWidget(self._ep, 3, 1)

        lay.addWidget(frame)
        btn = QPushButton("Resume Training")
        btn.setObjectName("primaryBtn")
        btn.clicked.connect(lambda: QMessageBox.information(
            self, "Coming Soon", "Backend will be connected in a future update."))
        lay.addWidget(btn)

    def _browse_file(self, attr, hint):
        f, _ = QFileDialog.getOpenFileName(self, f"Select {hint}", "", "All Files (*.*)")
        if f:
            getattr(self, attr).setText(f)

    def _browse_dir(self, attr):
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if d:
            getattr(self, attr).setText(d)


class RunComparisonDialog(QDialog):
    def __init__(self, app_state=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Comparison")
        self.setMinimumSize(700, 480)
        self.resize(820, 560)
        self._runs: list[str] = []
        self._build()
        auto_titlebar(self)

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        inner = QVBoxLayout(frame)
        inner.setSpacing(6)
        inner.setContentsMargins(10, 10, 10, 10)
        inner.addWidget(QLabel("Add results.csv files to compare:"))

        row = QHBoxLayout()
        self._csv_in = QLineEdit()
        self._csv_in.setPlaceholderText("results.csv path")
        self._csv_in.setReadOnly(True)
        row.addWidget(self._csv_in, stretch=1)
        b1 = QPushButton("Browse")
        b1.setFixedWidth(76)
        b1.clicked.connect(self._browse)
        row.addWidget(b1)
        b2 = QPushButton("Add Run")
        b2.setFixedWidth(76)
        b2.clicked.connect(self._add)
        row.addWidget(b2)
        inner.addLayout(row)

        self._run_list = QPlainTextEdit()
        self._run_list.setReadOnly(True)
        self._run_list.setMaximumHeight(70)
        self._run_list.setPlaceholderText("No runs added yet…")
        inner.addWidget(self._run_list)
        lay.addWidget(frame)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        clr = QPushButton("Clear All")
        clr.clicked.connect(self._clear)
        btn_row.addWidget(clr)
        cmp = QPushButton("Compare")
        cmp.setObjectName("primaryBtn")
        cmp.clicked.connect(lambda: QMessageBox.information(
            self, "Coming Soon", "Backend will be connected in a future update."))
        btn_row.addWidget(cmp)
        lay.addLayout(btn_row)

        ph = QLabel("Add runs and click Compare to overlay curves")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet("color: #888; font-size: 11pt;")
        ph.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(ph, stretch=1)

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select results.csv", "",
                                           "CSV Files (*.csv);;All Files (*.*)")
        if f:
            self._csv_in.setText(f)

    def _add(self):
        p = self._csv_in.text().strip()
        if p and p not in self._runs:
            self._runs.append(p)
            self._run_list.appendPlainText(f"  {len(self._runs)}. {p}")
        self._csv_in.clear()

    def _clear(self):
        self._runs.clear()
        self._run_list.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

class Sidebar(QWidget):
    model_changed = Signal(int)

    _MODEL_ITEMS = [
        ("Detection Nano",    "nano"),
        ("Detection Small",    "small"),
        ("Detection Medium",    "medium"),
        ("Detection Large",    "large"),
        ("Detection XLarge",    "xlarge"),
        ("Segmentation Nano", "nano"),
        ("Segmentation Small", "small"),
        ("Segmentation Medium", "medium"),
        ("Segmentation Large", "large"),
        ("Segmentation XLarge", "xlarge"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(10, 12, 10, 8)
        inner_lay.setSpacing(0)

        inner_lay.addWidget(_section_label("Configuration"))
        inner_lay.addSpacing(5)

        lbl_yaml = QLabel("Dataset YAML")
        lbl_yaml.setStyleSheet("font-size: 9pt; color: #706B63;")
        inner_lay.addWidget(lbl_yaml)
        inner_lay.addSpacing(2)

        row1 = QHBoxLayout()
        row1.setSpacing(3)
        self.dataset_linedit = QLineEdit()
        self.dataset_linedit.setPlaceholderText("dataset.yaml")
        row1.addWidget(self.dataset_linedit)
        self.dataset_button = QPushButton("…")
        self.dataset_button.setFixedSize(26, 26)
        self.dataset_button.setObjectName("iconBtn")
        row1.addWidget(self.dataset_button)
        inner_lay.addLayout(row1)
        inner_lay.addSpacing(8)

        lbl_out = QLabel("Output folder")
        lbl_out.setStyleSheet("font-size: 9pt; color: #706B63;")
        inner_lay.addWidget(lbl_out)
        inner_lay.addSpacing(2)

        row2 = QHBoxLayout()
        row2.setSpacing(3)
        self.output_linedit = QLineEdit()
        self.output_linedit.setPlaceholderText("output directory")
        row2.addWidget(self.output_linedit)
        self.output_button = QPushButton("…")
        self.output_button.setFixedSize(26, 26)
        self.output_button.setObjectName("iconBtn")
        row2.addWidget(self.output_button)
        inner_lay.addLayout(row2)
        inner_lay.addSpacing(12)

        inner_lay.addWidget(_hsep())
        inner_lay.addSpacing(10)

        inner_lay.addWidget(_section_label("Models"))
        inner_lay.addSpacing(4)
        self._model_list = QListWidget()
        self._model_list.setSpacing(0)
        self._model_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        for name, tag in self._MODEL_ITEMS:
            item = QListWidgetItem()
            item.setText(f"  {name}")
            item.setData(Qt.ItemDataRole.UserRole, tag)
            self._model_list.addItem(item)

        sep_item = QListWidgetItem("─" * 22)
        sep_item.setFlags(Qt.ItemFlag.NoItemFlags)
        sep_item.setForeground(QColor("#555"))
        self._model_list.insertItem(5, sep_item)
        seg_header = QListWidgetItem()
        seg_header.setFlags(Qt.ItemFlag.NoItemFlags)
        self._model_list.insertItem(6, seg_header)

        self._model_list.setCurrentRow(1)
        self._model_list.itemClicked.connect(self._on_model_click)
        self._model_list.setMaximumHeight(280)
        inner_lay.addWidget(self._model_list)
        inner_lay.addStretch()

        scroll.setWidget(inner)
        root.addWidget(scroll, stretch=1)

        # Footer — dataset tools
        footer = QWidget()
        footer.setFixedHeight(150)
        footer_lay = QVBoxLayout(footer)
        footer_lay.setContentsMargins(12, 6, 12, 8)
        footer_lay.setSpacing(4)

        footer_lay.addWidget(_hsep())
        footer_lay.addSpacing(0)
        footer_lay.addWidget(_section_label("Dataset tools"))

        links_col = QVBoxLayout()
        links_col.setSpacing(2)
        links_col.setContentsMargins(0, 0, 0, 0)

        self.analyze_btn     = _link_btn("Analyze")
        self.health_btn      = _link_btn("Health")
        self.organize_btn    = _link_btn("Organize")
        self.emptylabels_btn = _link_btn("Empty labels")

        for i, btn in enumerate([self.analyze_btn, self.health_btn,
                                  self.organize_btn, self.emptylabels_btn]):
            links_col.addWidget(btn)
            if i < 3:
                sep = _hsep()
                sep.setStyleSheet("background: #C4BFB5;")
                sep.setFixedHeight(2)
                sep.setContentsMargins(5, 0, 5, 0)
                links_col.addWidget(sep)

        footer_lay.addLayout(links_col)
        root.addWidget(footer)

    def _on_model_click(self, item: QListWidgetItem):
        if not (item.flags() & Qt.ItemFlag.ItemIsSelectable):
            return
        row = self._model_list.row(item)
        flat = row if row < 5 else row - 2
        self.model_changed.emit(flat)

    def current_model_index(self) -> int:
        row = self._model_list.currentRow()
        return row if row < 5 else max(0, row - 2)

    def set_model_index(self, idx: int):
        visual = idx if idx < 5 else idx + 2
        self._model_list.setCurrentRow(visual)


# ──────────────────────────────────────────────────────────────────────────────
# Train tab
# ──────────────────────────────────────────────────────────────────────────────

class TrainTab(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._build()

    def _build(self):
        container = QWidget()
        lay = QHBoxLayout(container)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)
        lay.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # Schedule
        sched = ParamCard("Schedule")
        self.resolution_spinbox = QSpinBox()
        self.resolution_spinbox.setRange(64, 2048)
        self.resolution_spinbox.setValue(640)
        self.resolution_spinbox.setSingleStep(32)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 2000)
        self.epochs_spinbox.setValue(100)
        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setRange(0, 500)
        self.patience_spinbox.setValue(30)
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 1024)
        self.batch_spinbox.setValue(16)
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setRange(0, 32)
        self.workers_spinbox.setValue(8)
        self.auto_batch_checkbox = QCheckBox("Auto batch size")
        for lbl, w in [
            ("Resolution", self.resolution_spinbox),
            ("Epochs",     self.epochs_spinbox),
            ("Patience",   self.patience_spinbox),
            ("Batch size", self.batch_spinbox),
            ("Workers",    self.workers_spinbox),
        ]:
            sched.add_row(ParamRow(lbl, w))
        sched.add_checkbox(self.auto_batch_checkbox)
        lay.addWidget(sched)

        # Regularization
        reg = ParamCard("Regularization")
        self.dropout_spinbox = QDoubleSpinBox()
        self.dropout_spinbox.setRange(0.0, 0.9)
        self.dropout_spinbox.setValue(0.0)
        self.dropout_spinbox.setSingleStep(0.05)
        self.dropout_spinbox.setDecimals(2)
        self.weight_decay_spinbox = QDoubleSpinBox()
        self.weight_decay_spinbox.setRange(0.0, 0.1)
        self.weight_decay_spinbox.setValue(0.0005)
        self.weight_decay_spinbox.setSingleStep(0.0001)
        self.weight_decay_spinbox.setDecimals(4)
        self.label_smoothing_spinbox = QDoubleSpinBox()
        self.label_smoothing_spinbox.setRange(0.0, 0.3)
        self.label_smoothing_spinbox.setValue(0.0)
        self.label_smoothing_spinbox.setSingleStep(0.01)
        self.label_smoothing_spinbox.setDecimals(2)
        self.warmup_epochs_spinbox = QDoubleSpinBox()
        self.warmup_epochs_spinbox.setRange(0.0, 10.0)
        self.warmup_epochs_spinbox.setValue(3.0)
        self.warmup_epochs_spinbox.setSingleStep(0.5)
        self.warmup_epochs_spinbox.setDecimals(1)
        self.cos_lr_checkbox = QCheckBox("Cosine LR schedule")
        for lbl, w, mn, mx in [
            ("Dropout",       self.dropout_spinbox,         0.0, 0.9),
            ("Weight decay",  self.weight_decay_spinbox,    0.0, 0.1),
            ("Label smooth",  self.label_smoothing_spinbox, 0.0, 0.3),
            ("Warmup epochs", self.warmup_epochs_spinbox,   0.0, 10.0),
        ]:
            reg.add_row(ParamRow(lbl, w, show_track=True, min_val=mn, max_val=mx))
        reg.add_checkbox(self.cos_lr_checkbox)
        lay.addWidget(reg)

        # Augmentation
        aug = ParamCard("Augmentation")
        self.mosaic_spinbox = QDoubleSpinBox()
        self.mosaic_spinbox.setRange(0.0, 1.0)
        self.mosaic_spinbox.setValue(1.0)
        self.mosaic_spinbox.setSingleStep(0.1)
        self.mosaic_spinbox.setDecimals(1)
        self.mixup_spinbox = QDoubleSpinBox()
        self.mixup_spinbox.setRange(0.0, 1.0)
        self.mixup_spinbox.setValue(0.0)
        self.mixup_spinbox.setSingleStep(0.1)
        self.mixup_spinbox.setDecimals(1)
        self.copy_paste_spinbox = QDoubleSpinBox()
        self.copy_paste_spinbox.setRange(0.0, 1.0)
        self.copy_paste_spinbox.setValue(0.0)
        self.copy_paste_spinbox.setSingleStep(0.1)
        self.copy_paste_spinbox.setDecimals(1)
        self.degrees_spinbox = QDoubleSpinBox()
        self.degrees_spinbox.setRange(0.0, 180.0)
        self.degrees_spinbox.setValue(0.0)
        self.degrees_spinbox.setSingleStep(5.0)
        self.degrees_spinbox.setDecimals(1)
        self.fliplr_spinbox = QDoubleSpinBox()
        self.fliplr_spinbox.setRange(0.0, 1.0)
        self.fliplr_spinbox.setValue(0.5)
        self.fliplr_spinbox.setSingleStep(0.1)
        self.fliplr_spinbox.setDecimals(1)
        self.flipud_spinbox = QDoubleSpinBox()
        self.flipud_spinbox.setRange(0.0, 1.0)
        self.flipud_spinbox.setValue(0.0)
        self.flipud_spinbox.setSingleStep(0.1)
        self.flipud_spinbox.setDecimals(1)
        self.hsv_h_spinbox = QDoubleSpinBox()
        self.hsv_h_spinbox.setRange(0.0, 1.0)
        self.hsv_h_spinbox.setValue(0.015)
        self.hsv_h_spinbox.setSingleStep(0.005)
        self.hsv_h_spinbox.setDecimals(3)
        self.hsv_s_spinbox = QDoubleSpinBox()
        self.hsv_s_spinbox.setRange(0.0, 1.0)
        self.hsv_s_spinbox.setValue(0.7)
        self.hsv_s_spinbox.setSingleStep(0.05)
        self.hsv_s_spinbox.setDecimals(2)
        self.hsv_v_spinbox = QDoubleSpinBox()
        self.hsv_v_spinbox.setRange(0.0, 1.0)
        self.hsv_v_spinbox.setValue(0.4)
        self.hsv_v_spinbox.setSingleStep(0.05)
        self.hsv_v_spinbox.setDecimals(2)
        for lbl, w, mn, mx in [
            ("Mosaic",      self.mosaic_spinbox,       0.0, 1.0),
            ("Mixup",       self.mixup_spinbox,         0.0, 1.0),
            ("Copy-paste",  self.copy_paste_spinbox,    0.0, 1.0),
            ("Rotation °",  self.degrees_spinbox,       0.0, 180.0),
            ("Flip LR",     self.fliplr_spinbox,        0.0, 1.0),
            ("Flip UD",     self.flipud_spinbox,         0.0, 1.0),
            ("HSV hue",     self.hsv_h_spinbox,         0.0, 1.0),
            ("HSV sat",     self.hsv_s_spinbox,         0.0, 1.0),
            ("HSV val",     self.hsv_v_spinbox,         0.0, 1.0),
        ]:
            aug.add_row(ParamRow(lbl, w, show_track=True, min_val=mn, max_val=mx))
        lay.addWidget(aug)

        # Last run
        last = ParamCard("Last run", accent_color="#2D7A4F")
        self._map50_lbl   = QLabel("—")
        self._map5095_lbl = QLabel("—")
        self._prec_lbl    = QLabel("—")
        self._rec_lbl     = QLabel("—")
        self._run_info    = QLabel("no run yet")

        for widget, style in [
            (self._map50_lbl,   "font-size: 22pt; font-weight: 500;"),
            (self._map5095_lbl, "font-size: 16pt; font-weight: 500;"),
        ]:
            widget.setStyleSheet(style + " background: transparent;")

        def _metric_block(big_lbl, caption):
            w = QWidget()
            l = QVBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            l.setSpacing(1)
            cap = QLabel(caption)
            cap.setStyleSheet("font-size: 8.5pt; color: #706B63; background: transparent;")
            l.addWidget(cap)
            l.addWidget(big_lbl)
            return w

        last.add_row(_metric_block(self._map50_lbl,   "mAP@50"))
        last.add_row(_metric_block(self._map5095_lbl, "mAP@50-95"))

        pair = QWidget()
        pair_lay = QHBoxLayout(pair)
        pair_lay.setContentsMargins(0, 0, 0, 0)
        pair_lay.setSpacing(12)
        pair_lay.addWidget(_metric_block(self._prec_lbl, "Precision"))
        pair_lay.addWidget(_metric_block(self._rec_lbl,  "Recall"))
        last.add_row(pair)

        self._run_info.setStyleSheet(
            "font-size: 8.5pt; color: #9E9C97; background: transparent;"
            "padding-top: 4px; border-top: 1px solid #D8D4CC;")
        last.add_row(self._run_info)

        last.setFixedWidth(180)
        lay.addWidget(last)
        lay.addStretch()

        self.setWidget(container)

    def update_last_run(self, map50: float, map5095: float,
                        prec: float, rec: float, info: str):
        self._map50_lbl.setText(f"{map50:.3f}")
        self._map5095_lbl.setText(f"{map5095:.3f}")
        self._prec_lbl.setText(f"{prec:.3f}")
        self._rec_lbl.setText(f"{rec:.3f}")
        self._run_info.setText(info)


# ──────────────────────────────────────────────────────────────────────────────
# Curves tab
# ──────────────────────────────────────────────────────────────────────────────

class CurvesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        row = QHBoxLayout(frame)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)
        row.addWidget(QLabel("results.csv:"))
        self._csv_in = QLineEdit()
        self._csv_in.setPlaceholderText("Select a YOLO results.csv — or finish a training run")
        self._csv_in.setReadOnly(True)
        row.addWidget(self._csv_in, stretch=1)
        b = QPushButton("Browse")
        b.setFixedWidth(76)
        b.clicked.connect(self._browse)
        row.addWidget(b)
        load = QPushButton("Load")
        load.setObjectName("primaryBtn")
        load.setFixedWidth(64)
        load.clicked.connect(lambda: QMessageBox.information(
            self, "Coming Soon", "Curve rendering will be connected in a future update."))
        row.addWidget(load)
        lay.addWidget(frame)

        ph = QLabel("Load a results.csv to display training curves")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet("color: #888; font-size: 12pt;")
        ph.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(ph, stretch=1)

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select results.csv", "",
                                           "CSV (*.csv);;All Files (*.*)")
        if f:
            self._csv_in.setText(f)

    def load_csv(self, path: str):
        self._csv_in.setText(path)


# ──────────────────────────────────────────────────────────────────────────────
# Health tab
# ──────────────────────────────────────────────────────────────────────────────

class HealthTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        g = QGridLayout(frame)
        g.setContentsMargins(10, 8, 10, 8)
        g.setSpacing(8)
        g.setColumnStretch(1, 1)
        g.addWidget(QLabel("Dataset folder:"), 0, 0)
        self._folder = QLineEdit()
        self._folder.setPlaceholderText("Root dataset folder to inspect")
        self._folder.setReadOnly(True)
        g.addWidget(self._folder, 0, 1)
        b = QPushButton("Browse")
        b.setFixedWidth(76)
        b.clicked.connect(self._browse)
        g.addWidget(b, 0, 2)
        lay.addWidget(frame)

        run_btn = QPushButton("Run Health Check")
        run_btn.setObjectName("primaryBtn")
        run_btn.clicked.connect(lambda: QMessageBox.information(
            self, "Coming Soon", "Health check backend will be connected in a future update."))
        lay.addWidget(run_btn)

        self._results = QPlainTextEdit()
        self._results.setReadOnly(True)
        self._results.setPlaceholderText(
            "Health check results will appear here…\n\n"
            "Checks include:\n"
            "  •  Classes with fewer than 30 samples\n"
            "  •  Images missing label files\n"
            "  •  Label files with out-of-range coordinates\n"
            "  •  Severe class imbalance (>10× ratio)\n"
            "  •  Duplicate filenames across splits"
        )
        lay.addWidget(self._results, stretch=1)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if d:
            self._folder.setText(d)


# ──────────────────────────────────────────────────────────────────────────────
# ONNX / HEF tab
# ──────────────────────────────────────────────────────────────────────────────

class OnnxTab(QWidget):
    def __init__(self, app_state=None, parent=None):
        super().__init__(parent)
        self._state = app_state
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        src = QFrame()
        src.setFrameShape(QFrame.Shape.StyledPanel)
        sg = QGridLayout(src)
        sg.setContentsMargins(10, 10, 10, 10)
        sg.setSpacing(8)
        sg.setColumnStretch(1, 1)
        src_title = QLabel("Source")
        src_title.setStyleSheet("font-size: 9.5pt; font-weight: 600;")
        sg.addWidget(src_title, 0, 0, 1, 3)

        def _path_row(grid, row, label, attr, ph, browse_fn):
            grid.addWidget(QLabel(label), row, 0)
            le = QLineEdit()
            le.setPlaceholderText(ph)
            le.setReadOnly(True)
            setattr(self, attr, le)
            grid.addWidget(le, row, 1)
            b = QPushButton("…")
            b.setObjectName("iconBtn")
            b.setFixedSize(26, 26)
            b.clicked.connect(browse_fn)
            grid.addWidget(b, row, 2)

        _path_row(sg, 1, "ONNX file:", "onnx_input", "train/weights/best.onnx",
                  lambda: self._browse_file("onnx_input", "ONNX (*.onnx)"))
        _path_row(sg, 2, "Dataset YAML:", "yaml_input", "dataset.yaml",
                  lambda: self._browse_file("yaml_input", "YAML (*.yaml *.yml)"))
        _path_row(sg, 3, "Output folder:", "out_input", "export output",
                  lambda: self._browse_dir("out_input"))
        lay.addWidget(src)

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

        export_btn = QPushButton("Export Model")
        export_btn.setObjectName("primaryBtn")
        export_btn.clicked.connect(self._open_exporter)
        lay.addWidget(export_btn)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(2000)
        self.log_box.setPlaceholderText("Export logs will appear here…")
        lay.addWidget(self.log_box, stretch=1)

    def _browse_file(self, attr: str, filt: str):
        f, _ = QFileDialog.getOpenFileName(self, "Select file", "",
                                           f"{filt};;All Files (*.*)")
        if f:
            getattr(self, attr).setText(f)

    def _browse_dir(self, attr: str):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            getattr(self, attr).setText(d)

    def _open_exporter(self):
        dlg = Exporter(app_state=self._state)
        if self.onnx_input.text():
            dlg.onnx_path_input.setText(self.onnx_input.text())
        if self.yaml_input.text():
            dlg.yaml_file_input.setText(self.yaml_input.text())
        if self.out_input.text():
            dlg.output_input.setText(self.out_input.text())
        dlg.resolution_input.setValue(self.resolution_input.value())
        if self.model_name_input.text():
            dlg.model_name_input.setText(self.model_name_input.text())
        dlg.exec()


# ──────────────────────────────────────────────────────────────────────────────
# Title bar
# ──────────────────────────────────────────────────────────────────────────────

class TitleBar(QWidget):
    labelme_clicked     = Signal()
    organize_clicked    = Signal()
    emptylabels_clicked = Signal()
    resume_clicked      = Signal()
    compare_clicked     = Signal()
    theme_toggled       = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(38)
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 10, 0)
        lay.setSpacing(4)

        title = QLabel("TRAINR")
        title.setStyleSheet(
            "font-size: 11pt; font-weight: 700; letter-spacing: 0.06em;"
            "background: transparent;"
        )
        lay.addWidget(title)
        lay.addStretch()

        self._lm_btn  = _icon_btn("🏷",  "Open LabelMe")
        self._org_btn = _icon_btn("📁",  "Organize dataset")
        self._emp_btn = _icon_btn("📄",  "Generate empty labels")
        self._res_btn = _icon_btn("▶▶", "Resume training")
        self._cmp_btn = _icon_btn("📊", "Compare runs")

        self._lm_btn.clicked.connect(self.labelme_clicked)
        self._org_btn.clicked.connect(self.organize_clicked)
        self._emp_btn.clicked.connect(self.emptylabels_clicked)
        self._res_btn.clicked.connect(self.resume_clicked)
        self._cmp_btn.clicked.connect(self.compare_clicked)

        for btn in [self._lm_btn, self._org_btn, self._emp_btn,
                    self._res_btn, self._cmp_btn]:
            lay.addWidget(btn)

        sep = _vsep()
        sep.setFixedHeight(16)
        lay.addWidget(sep)

        self._theme_btn = QPushButton("☀ Light")
        self._theme_btn.setObjectName("iconBtn")
        self._theme_btn.setFixedHeight(26)
        self._theme_btn.setStyleSheet(
            self._theme_btn.styleSheet() + "padding: 0 8px; font-size: 9pt;"
        )
        self._theme_btn.clicked.connect(self.theme_toggled)
        lay.addWidget(self._theme_btn)

    def set_theme_label(self, theme: str):
        self._theme_btn.setText("🌙 Dark" if theme == "light" else "☀ Light")

    def paintEvent(self, event):
        p = QPainter(self)
        pal = palette()
        p.setPen(QPen(QColor(pal["BORDER_S"]), 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()


# ──────────────────────────────────────────────────────────────────────────────
# Status strip  (replaces the old QStatusBar)
# ──────────────────────────────────────────────────────────────────────────────

class StatusStrip(QWidget):
    """
    Slim accent-coloured strip at the very bottom of the window.
    Left  : small status indicator dot + current operation text
    Centre: epoch progress pill (hidden when idle)
    Right : START TRAINING button — styled to sit flush in the strip
    """

    start_clicked = Signal()

    # State constants
    IDLE     = "idle"
    TRAINING = "training"
    EXPORTING= "exporting"
    DONE     = "done"
    FAILED   = "failed"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self._state    = self.IDLE
        self._build()
        self._set_state(self.IDLE)

    # ── build ──────────────────────────────────────────────────────────────

    def _build(self):
        pal = palette()
        self.setStyleSheet(f"background: {pal['STATUS_BG']};")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self._btn_container = QWidget()
        btn_container = self._btn_container        
        btn_container.setStyleSheet("background: rgba(0,0,0,0.12); border-radius: 0;")
        btn_container_lay = QHBoxLayout(btn_container)
        btn_container_lay.setContentsMargins(8, 0, 0, 0)
        lay.addWidget(btn_container)
        # Indicator dot
        self._dot = QLabel("●")
        self._dot.setFixedWidth(12)
        self._dot.setStyleSheet(
            "font-size: 7pt; color: rgba(255,255,255,0.5);"
            "background: transparent;"
        )
        btn_container_lay.addWidget(self._dot)

        # Status text
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(
            "font-size: 9pt; color: rgba(255,255,255,0.82);"
            "background: transparent;"
        )
        btn_container_lay.addWidget(self._status_lbl)

        # Epoch pill — hidden until training starts
        self._epoch_pill = QLabel("")
        self._epoch_pill.setStyleSheet(
            "font-size: 8.5pt; font-variant-numeric: tabular-nums;"
            "color: rgba(255,255,255,0.7);"
            "background: rgba(0,0,0,0.18);"
            "border-radius: 3px; padding: 1px 7px;"
            "background: transparent;"
        )
        self._epoch_pill.setVisible(False)
        btn_container_lay.addWidget(self._epoch_pill)

        btn_container_lay.addStretch()

        # START TRAINING button — flat, lives in the strip
        self.start_btn = QPushButton("▶  Start Training")
        self.start_btn.setFixedHeight(22)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.14);
                color: #fff;
                border: 1px solid rgba(255,255,255,0.28);
                border-radius: 4px;
                font-size: 9pt;
                font-weight: 600;
                letter-spacing: 0.04em;
                padding: 0 16px;
            }
            QPushButton:hover {
                background: rgba(255,255,255,0.24);
                border-color: rgba(255,255,255,0.45);
            }
            QPushButton:pressed {
                background: rgba(0,0,0,0.12);
            }
            QPushButton:disabled {
                color: rgba(255,255,255,0.35);
                border-color: rgba(255,255,255,0.12);
                background: rgba(0,0,0,0.08);
            }
        """)
        self.start_btn.clicked.connect(self.start_clicked)
        btn_container_lay.addWidget(self.start_btn)
        self.refresh_color()

    # ── public API ─────────────────────────────────────────────────────────

    def set_idle(self):
        self._set_state(self.IDLE)
        self._epoch_pill.setVisible(False)
        self._epoch_pill.setText("")

    def set_training(self, model_name: str = ""):
        self._set_state(self.TRAINING)
        msg = f"Training  {model_name}" if model_name else "Training"
        self._status_lbl.setText(msg)
        self.start_btn.setEnabled(False)

    def set_epoch(self, current: int, total: int):
        """Update epoch counter without changing state."""
        self._epoch_pill.setText(f"epoch {current} / {total}")
        self._epoch_pill.setVisible(True)

    def set_exporting(self):
        self._set_state(self.EXPORTING)
        self._epoch_pill.setVisible(False)

    def set_done(self):
        self._set_state(self.DONE)
        self._epoch_pill.setVisible(False)
        self.start_btn.setEnabled(True)

    def set_failed(self, job: str = ""):
        self._set_state(self.FAILED)
        if job:
            self._status_lbl.setText(f"Failed  ·  {job}")
        self._epoch_pill.setVisible(False)
        self.start_btn.setEnabled(True)

    def refresh_color(self):
        pal = palette()
        self.setStyleSheet(f"background: {pal['STATUS_BG']};")
        self._btn_container.setStyleSheet("background: rgba(0,0,0,0.10); border-radius: 0;")
        
        self._status_lbl.setStyleSheet(
            f"font-size: 9pt; color: {pal['TEXT']}; background: transparent;")
        self._epoch_pill.setStyleSheet(
            f"font-size: 8.5pt; font-variant-numeric: tabular-nums;"
            f"color: {pal['TEXT_2']}; background: transparent;")
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(0,0,0,0.12);
                color: {pal['TEXT']};
                border: 1px solid rgba(0,0,0,0.15);
                border-radius: 4px;
                font-size: 9pt;
                font-weight: 600;
                letter-spacing: 0.04em;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background: rgba(0,0,0,0.20);
            }}
            QPushButton:pressed {{
                background: rgba(0,0,0,0.28);
            }}
            QPushButton:disabled {{
                color: {pal['TEXT_3']};
                background: rgba(0,0,0,0.06);
            }}
        """)
        self._set_state(self._state)

    # ── internal ───────────────────────────────────────────────────────────

    _STATE_CFG = {
        IDLE:      ("rgba(255,255,255,0.35)", "Ready"),
        TRAINING:  ("#8DF5B0",               "Training"),   # text set by set_training
        EXPORTING: ("rgba(255,255,255,0.55)", "Exporting ONNX…"),
        DONE:      ("#8DF5B0",               "Complete"),
        FAILED:    ("#FF8A80",               "Failed"),
    }

    def _set_state(self, state: str):
        self._state = state
        dot_color, status_text = self._STATE_CFG.get(
            state, ("rgba(255,255,255,0.35)", ""))
        self._dot.setStyleSheet(
            f"font-size: 7pt; color: {dot_color}; background: transparent;")
        if state not in (self.TRAINING,):   # training label set externally
            self._status_lbl.setText(status_text)


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────

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
        self.current_job  = None
        self._epoch_total = 0

        self.setWindowTitle("TRAINR")
        self.resize(1150, 680)
        self.setMinimumSize(860, 580)

        self._build_ui()
        self._connect_signals()

        if self.state:
            self.load_state()
            self.bind_state()

    # ──────────────────────────────────────────────────────────────────────────
    # Build
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
        main_col = QWidget()
        main_col_lay = QVBoxLayout(main_col)
        main_col_lay.setContentsMargins(0, 0, 0, 0)
        main_col_lay.setSpacing(0)

        self.tab_bar = TabBar(["Train", "Curves", "Health", "ONNX / HEF"])
        main_col_lay.addWidget(self.tab_bar)

        tb_sep = QFrame()
        tb_sep.setFrameShape(QFrame.Shape.HLine)
        tb_sep.setFixedHeight(1)
        main_col_lay.addWidget(tb_sep)

        self.pages = QStackedWidget()
        self.train_tab  = TrainTab()
        self.curves_tab = CurvesTab()
        self.health_tab = HealthTab()
        self.onnx_tab   = OnnxTab(app_state=self.state)
        self.pages.addWidget(self.train_tab)
        self.pages.addWidget(self.curves_tab)
        self.pages.addWidget(self.health_tab)
        self.pages.addWidget(self.onnx_tab)
        main_col_lay.addWidget(self.pages, stretch=1)

        # Log panel
        log_panel = QWidget()
        log_panel_lay = QVBoxLayout(log_panel)
        log_panel_lay.setContentsMargins(0, 0, 0, 0)
        log_panel_lay.setSpacing(0)

        log_top = QWidget()
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

        log_sep = QFrame()
        log_sep.setFrameShape(QFrame.Shape.HLine)
        log_sep.setFixedHeight(1)
        log_panel_lay.addWidget(log_sep)

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

        # ── Status strip (replaces QStatusBar) ────────────────────────────
        self.status_strip = StatusStrip()
        root.addWidget(self.status_strip)

    # ──────────────────────────────────────────────────────────────────────────
    # Signals
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.tab_bar.tab_changed.connect(self.pages.setCurrentIndex)

        self.sidebar.dataset_button.clicked.connect(self._browse_yaml)
        self.sidebar.output_button.clicked.connect(self._browse_output)

        self.sidebar.analyze_btn.clicked.connect(lambda: DatasetVisualizer().exec())
        self.sidebar.health_btn.clicked.connect(
            lambda: (self.pages.setCurrentIndex(2), self.tab_bar._select(2)))
        self.sidebar.organize_btn.clicked.connect(
            lambda: OrganizerWindow(app_state=self.state).exec())
        self.sidebar.emptylabels_btn.clicked.connect(lambda: EmptyLabelsDialog().exec())

        self.titlebar.labelme_clicked.connect(
            lambda: QProcess.startDetached(str(LABELME)))
        self.titlebar.organize_clicked.connect(
            lambda: OrganizerWindow(app_state=self.state).exec())
        self.titlebar.emptylabels_clicked.connect(lambda: EmptyLabelsDialog().exec())
        self.titlebar.resume_clicked.connect(
            lambda: ResumeTrainingDialog(app_state=self.state, parent=self).exec())
        self.titlebar.compare_clicked.connect(
            lambda: RunComparisonDialog(app_state=self.state, parent=self).exec())
        self.titlebar.theme_toggled.connect(self._toggle_theme)

        self.status_strip.start_clicked.connect(self.start_training)

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
    # State persistence
    # ──────────────────────────────────────────────────────────────────────────

    def load_state(self):
        s = self.state
        t = self.train_tab
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

    def bind_state(self):
        s = self.state
        t = self.train_tab
        sb = self.sidebar
        sb.dataset_linedit.textChanged.connect(lambda v: s.set("trainr.dataset", v))
        sb.output_linedit.textChanged.connect(lambda v: s.set("trainr.output", v))
        sb.model_changed.connect(lambda v: s.set("trainr.model", v))
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
                "ERROR: YOLO CLI not found. Please run the Heavy Installer first.")
            return

        dataset = self.sidebar.dataset_linedit.text().strip()
        output  = self.sidebar.output_linedit.text().strip()

        if not dataset or not output:
            self.log_box.appendPlainText("ERROR: Dataset or output path missing.")
            return

        model_idx  = self.sidebar.current_model_index()
        model_name = self._MODEL_MAP.get(model_idx, "yolov8s.pt")
        task       = "segment" if "-seg" in model_name else "detect"

        t     = self.train_tab
        batch = "-1" if t.auto_batch_checkbox.isChecked() else str(t.batch_spinbox.value())

        cmd = [
            str(YOLO_EXE), task, "train",
            f"data={dataset}",
            f"model={MODELS / model_name}",
            f"imgsz={t.resolution_spinbox.value()}",
            f"epochs={t.epochs_spinbox.value()}",
            f"batch={batch}",
            f"patience={t.patience_spinbox.value()}",
            f"workers={t.workers_spinbox.value()}",
            f"project={output}",
            "name=train", "exist_ok=True",
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

        self._epoch_total = t.epochs_spinbox.value()
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

        # Feed epoch progress to the status strip
        for line in raw.splitlines():
            if "/" not in line:
                continue
            for token in line.split():
                if "/" in token:
                    try:
                        cur_s, tot_s = token.split("/", 1)
                        cur, tot = int(cur_s), int(tot_s)
                        self.status_strip.set_epoch(cur, tot)
                        return          # one update per chunk is enough
                    except ValueError:
                        pass

    def _process_finished(self, exit_code, exit_status):
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
            results_csv = (
                Path(self.sidebar.output_linedit.text()) / "train" / "results.csv"
            )
            if results_csv.exists():
                self.curves_tab.load_csv(str(results_csv))
                self.tab_bar._select(1)

    def _export_onnx(self):
        output  = self.sidebar.output_linedit.text().strip()
        weights = f"{output}/train/weights/best.pt"
        cmd = [
            str(YOLO_EXE), "export",
            f"model={weights}",
            "format=onnx", "opset=11", "simplify=True",
        ]
        self.log_box.appendPlainText("\nExporting ONNX…\n")
        self._run_process(cmd)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

app   = QApplication(sys.argv)
apply_theme(app, "dark")
state = AppState(str(CONFIG))

window = MainWindow(state)
window.show()
auto_titlebar(window)

app.exec()
