"""
ui/train_tab.py  —  TRAINR
TrainTab  : scrollable parameter cards (Schedule / Regularization / Augmentation / Last run)
ParamCard : styled card widget with dot-header
ParamRow  : label + optional track bar + spinbox
_TrackBar : QPainter-drawn mini progress bar
"""
from __future__ import annotations

from PySide6.QtCore    import Qt
from PySide6.QtGui     import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFrame, QHBoxLayout,
    QLabel, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from theme import palette


# ──────────────────────────────────────────────────────────────────────────────
# _TrackBar
# ──────────────────────────────────────────────────────────────────────────────

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
        w, h      = self.width(), self.height()
        cy        = h // 2
        groove_h  = 3
        p.setPen(Qt.PenStyle.NoPen)
        pal = palette()
        p.setBrush(QColor(pal["BORDER_S"]))
        p.drawRoundedRect(0, cy - groove_h // 2, w, groove_h, 1, 1)
        rng    = self._max - self._min or 1
        ratio  = max(0.0, min(1.0, (self._spin.value() - self._min) / rng))
        fill_w = int(w * ratio)
        if fill_w > 0:
            p.setBrush(QColor(pal["ACCENT"]))
            p.drawRoundedRect(0, cy - groove_h // 2, fill_w, groove_h, 1, 1)
        p.end()


# ──────────────────────────────────────────────────────────────────────────────
# ParamRow
# ──────────────────────────────────────────────────────────────────────────────

class ParamRow(QWidget):
    def __init__(self, label: str, spinbox: QWidget,
                 show_track: bool = False,
                 min_val: float = 0.0, max_val: float = 1.0,
                 parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        lbl = QLabel(label)
        lbl.setStyleSheet("font-size: 9.5pt; color: #706B63;")
        lbl.setMinimumWidth(90)
        layout.addWidget(lbl)

        if show_track:
            track = _TrackBar(spinbox, min_val, max_val)
            track.setFixedHeight(14)
            layout.addWidget(track, stretch=1)

        layout.addWidget(spinbox)
        spinbox.setFixedWidth(72)

    def value(self):
        return self._spinbox.value()   # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────────────────────
# ParamCard
# ──────────────────────────────────────────────────────────────────────────────

class ParamCard(QFrame):
    def __init__(self, title: str, accent_color: str | None = None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("paramCard")

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
            "background: transparent;")
        hdr_lay.addWidget(dot)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            "font-size: 9.5pt; font-weight: 600; background: transparent;")
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
# TrainTab
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

        # ── Schedule ──────────────────────────────────────────────────────────
        sched = ParamCard("Schedule")
        self.resolution_spinbox = _spin(64, 2048, 640, step=32)
        self.epochs_spinbox     = _spin(1, 2000, 100)
        self.patience_spinbox   = _spin(0, 500, 30)
        self.batch_spinbox      = _spin(1, 1024, 16)
        self.workers_spinbox    = _spin(0, 32, 8)
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

        # ── Regularization ────────────────────────────────────────────────────
        reg = ParamCard("Regularization")
        self.dropout_spinbox        = _dspin(0.0, 0.9,   0.0,    0.05, 2)
        self.weight_decay_spinbox   = _dspin(0.0, 0.1,   0.0005, 0.0001, 4)
        self.label_smoothing_spinbox= _dspin(0.0, 0.3,   0.0,    0.01, 2)
        self.warmup_epochs_spinbox  = _dspin(0.0, 10.0,  3.0,    0.5,  1)
        self.cos_lr_checkbox        = QCheckBox("Cosine LR schedule")
        for lbl, w, mn, mx in [
            ("Dropout",       self.dropout_spinbox,         0.0, 0.9),
            ("Weight decay",  self.weight_decay_spinbox,    0.0, 0.1),
            ("Label smooth",  self.label_smoothing_spinbox, 0.0, 0.3),
            ("Warmup epochs", self.warmup_epochs_spinbox,   0.0, 10.0),
        ]:
            reg.add_row(ParamRow(lbl, w, show_track=True, min_val=mn, max_val=mx))
        reg.add_checkbox(self.cos_lr_checkbox)
        lay.addWidget(reg)

        # ── Augmentation ──────────────────────────────────────────────────────
        aug = ParamCard("Augmentation")
        self.mosaic_spinbox      = _dspin(0.0, 1.0,   1.0,   0.1,   1)
        self.mixup_spinbox       = _dspin(0.0, 1.0,   0.0,   0.1,   1)
        self.copy_paste_spinbox  = _dspin(0.0, 1.0,   0.0,   0.1,   1)
        self.degrees_spinbox     = _dspin(0.0, 180.0, 0.0,   5.0,   1)
        self.fliplr_spinbox      = _dspin(0.0, 1.0,   0.5,   0.1,   1)
        self.flipud_spinbox      = _dspin(0.0, 1.0,   0.0,   0.1,   1)
        self.hsv_h_spinbox       = _dspin(0.0, 1.0,   0.015, 0.005, 3)
        self.hsv_s_spinbox       = _dspin(0.0, 1.0,   0.7,   0.05,  2)
        self.hsv_v_spinbox       = _dspin(0.0, 1.0,   0.4,   0.05,  2)
        for lbl, w, mn, mx in [
            ("Mosaic",     self.mosaic_spinbox,     0.0, 1.0),
            ("Mixup",      self.mixup_spinbox,      0.0, 1.0),
            ("Copy-paste", self.copy_paste_spinbox, 0.0, 1.0),
            ("Rotation °", self.degrees_spinbox,    0.0, 180.0),
            ("Flip LR",    self.fliplr_spinbox,     0.0, 1.0),
            ("Flip UD",    self.flipud_spinbox,     0.0, 1.0),
            ("HSV hue",    self.hsv_h_spinbox,      0.0, 1.0),
            ("HSV sat",    self.hsv_s_spinbox,      0.0, 1.0),
            ("HSV val",    self.hsv_v_spinbox,      0.0, 1.0),
        ]:
            aug.add_row(ParamRow(lbl, w, show_track=True, min_val=mn, max_val=mx))
        lay.addWidget(aug)

        # ── Last run ──────────────────────────────────────────────────────────
        last = ParamCard("Last run", accent_color="#2D7A4F")
        self._map50_lbl   = QLabel("—")
        self._map5095_lbl = QLabel("—")
        self._prec_lbl    = QLabel("—")
        self._rec_lbl     = QLabel("—")
        self._run_info    = QLabel("no run yet")

        for w, style in [
            (self._map50_lbl,   "font-size: 22pt; font-weight: 500;"),
            (self._map5095_lbl, "font-size: 16pt; font-weight: 500;"),
        ]:
            w.setStyleSheet(style + " background: transparent;")

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


# ── private helpers ───────────────────────────────────────────────────────────

def _spin(lo: int, hi: int, val: int, step: int = 1) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    return s


def _dspin(lo: float, hi: float, val: float,
           step: float, dec: int) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(dec)
    return s


def _metric_block(big_lbl: QLabel, caption: str) -> QWidget:
    w = QWidget()
    l = QVBoxLayout(w)
    l.setContentsMargins(0, 0, 0, 0)
    l.setSpacing(1)
    cap = QLabel(caption)
    cap.setStyleSheet("font-size: 8.5pt; color: #706B63; background: transparent;")
    l.addWidget(cap)
    l.addWidget(big_lbl)
    return w
