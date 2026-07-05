"""
ui/titlebar.py  —  TRAINR
TitleBar  : app name + icon utility buttons + theme toggle
StatusStrip: accent-coloured bottom strip with state machine + Start button
"""
from __future__ import annotations

from PySide6.QtCore  import Qt, Signal
from PySide6.QtGui   import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QWidget,
)

from theme import palette
from ui.widgets import icon_btn, vsep


# ──────────────────────────────────────────────────────────────────────────────
# TitleBar
# ──────────────────────────────────────────────────────────────────────────────

class TitleBar(QWidget):
    labelme_clicked  = Signal()
    language_clicked = Signal()
    reset_clicked    = Signal()
    resume_clicked   = Signal()
    compare_clicked  = Signal()
    theme_toggled    = Signal()

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
            "background: transparent;")
        lay.addWidget(title)
        lay.addStretch()

        self._lm_btn    = icon_btn("🏷",  "Open LabelMe")
        self._lang_btn  = icon_btn("🌐",  "Language")
        self._reset_btn = icon_btn("↺",   "Reset parameters to defaults")
        self._res_btn   = icon_btn("▶▶", "Resume training")
        self._cmp_btn   = icon_btn("📊",  "Compare runs")

        self._lm_btn.clicked.connect(self.labelme_clicked)
        self._lang_btn.clicked.connect(self.language_clicked)
        self._reset_btn.clicked.connect(self.reset_clicked)
        self._res_btn.clicked.connect(self.resume_clicked)
        self._cmp_btn.clicked.connect(self.compare_clicked)

        for btn in [self._lm_btn, self._lang_btn, self._reset_btn,
                    self._res_btn, self._cmp_btn]:
            lay.addWidget(btn)

        sep = vsep()
        sep.setFixedHeight(16)
        lay.addWidget(sep)

        self._theme_btn = QPushButton("☀ Light")
        self._theme_btn.setObjectName("iconBtn")
        self._theme_btn.setFixedHeight(26)
        self._theme_btn.setStyleSheet(
            self._theme_btn.styleSheet() + "padding: 0 8px; font-size: 9pt;")
        self._theme_btn.clicked.connect(self.theme_toggled)
        lay.addWidget(self._theme_btn)

    def set_theme_label(self, theme: str):
        self._theme_btn.setText("🌙 Dark" if theme == "light" else "☀ Light")

    def paintEvent(self, _event):
        p = QPainter(self)
        pal = palette()
        p.setPen(QPen(QColor(pal["BORDER_S"]), 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()


# ──────────────────────────────────────────────────────────────────────────────
# StatusStrip
# ──────────────────────────────────────────────────────────────────────────────

class StatusStrip(QWidget):
    start_clicked = Signal()

    IDLE      = "idle"
    TRAINING  = "training"
    EXPORTING = "exporting"
    DONE      = "done"
    FAILED    = "failed"

    _STATE_CFG = {
        "idle":      ("rgba(255,255,255,0.35)", "Ready"),
        "training":  ("#8DF5B0",               "Training"),
        "exporting": ("rgba(255,255,255,0.55)", "Exporting ONNX…"),
        "done":      ("#8DF5B0",               "Complete"),
        "failed":    ("#FF8A80",               "Failed"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self._state = self.IDLE
        self._build()
        self._set_state(self.IDLE)

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self._btn_container = QWidget()
        self._btn_container.setStyleSheet(
            "background: rgba(0,0,0,0.12); border-radius: 0;")
        cl = QHBoxLayout(self._btn_container)
        cl.setContentsMargins(8, 0, 0, 0)
        lay.addWidget(self._btn_container)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(12)
        cl.addWidget(self._dot)

        self._status_lbl = QLabel("Ready")
        cl.addWidget(self._status_lbl)

        self._epoch_pill = QLabel("")
        self._epoch_pill.setVisible(False)
        cl.addWidget(self._epoch_pill)

        cl.addStretch()

        self.start_btn = QPushButton("▶  Start Training")
        self.start_btn.setFixedHeight(22)
        self.start_btn.clicked.connect(self.start_clicked)
        cl.addWidget(self.start_btn)

        self.refresh_color()

    # ── public API ────────────────────────────────────────────────────────────

    def set_idle(self):
        self._set_state(self.IDLE)
        self._epoch_pill.setVisible(False)
        self._epoch_pill.setText("")

    def set_training(self, model_name: str = ""):
        self._set_state(self.TRAINING)
        self._status_lbl.setText(
            f"Training  {model_name}" if model_name else "Training")
        self.start_btn.setEnabled(False)

    def set_epoch(self, current: int, total: int):
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
        self._btn_container.setStyleSheet(
            "background: rgba(0,0,0,0.10); border-radius: 0;")
        self._status_lbl.setStyleSheet(
            f"font-size: 9pt; color: {pal['TEXT']}; background: transparent;")
        self._epoch_pill.setStyleSheet(
            f"font-size: 8.5pt; font-variant-numeric: tabular-nums;"
            f"color: {pal['TEXT_2']}; background: transparent;")
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(0,0,0,0.12); color: {pal['TEXT']};
                border: 1px solid rgba(0,0,0,0.15); border-radius: 4px;
                font-size: 9pt; font-weight: 600;
                letter-spacing: 0.04em; padding: 0 16px;
            }}
            QPushButton:hover   {{ background: rgba(0,0,0,0.20); }}
            QPushButton:pressed {{ background: rgba(0,0,0,0.28); }}
            QPushButton:disabled {{ color: {pal['TEXT_3']}; background: rgba(0,0,0,0.06); }}
        """)
        self._set_state(self._state)

    # ── internal ──────────────────────────────────────────────────────────────

    def _set_state(self, state: str):
        self._state = state
        dot_color, text = self._STATE_CFG.get(
            state, ("rgba(255,255,255,0.35)", ""))
        self._dot.setStyleSheet(
            f"font-size: 7pt; color: {dot_color}; background: transparent;")
        if state != self.TRAINING:
            self._status_lbl.setText(text)
