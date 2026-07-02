"""
ui/widgets.py  —  TRAINR
Shared small widgets and factory helpers used across the UI.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui  import QColor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget,
)

from theme import palette


# ──────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────────────────────

def icon_btn(char: str, tooltip: str) -> QPushButton:
    b = QPushButton(char)
    b.setObjectName("iconBtn")
    b.setToolTip(tooltip)
    b.setFixedSize(28, 28)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def link_btn(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("linkBtn")
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    return b


def section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        "font-size: 8pt; font-weight: 700; letter-spacing: 0.09em; color: #9E9C97;")
    return lbl


def hsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    return f


def vsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setFixedWidth(1)
    return f


# ──────────────────────────────────────────────────────────────────────────────
# TabBar
# ──────────────────────────────────────────────────────────────────────────────

class TabBar(QWidget):
    tab_changed = Signal(int)

    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
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
            lay.addWidget(btn, stretch=1)

        self._apply_styles()
        self._btns[0].setChecked(True)

    def _apply_styles(self):
        pal = palette()
        for b in self._btns:
            b.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: none;
                    border-bottom: 2px solid transparent; border-radius: 0;
                    padding: 6px 14px; font-size: 10pt;
                    color: {pal['TEXT_3']}; min-height: 0;
                }}
                QPushButton:checked {{
                    color: {pal['TEXT']};
                    border-bottom: 2px solid {pal['ACCENT']};
                    font-weight: 600;
                }}
                QPushButton:hover:!checked {{ color: {pal['TEXT_2']}; }}
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
