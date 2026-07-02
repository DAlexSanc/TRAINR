"""
theme.py  —  TRAINR
Two palettes: "dark" (default) and "light" (silver / off-white / burnt-orange).
"""
from __future__ import annotations
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

# ── Dark palette ──────────────────────────────────────────────────────────────
_D = dict(
    BG        = "#1A1A1A",
    SURFACE   = "#222222",
    SURFACE2  = "#2A2A2A",
    SURFACE3  = "#323232",
    BORDER    = "#333333",
    BORDER_S  = "#404040",
    ACCENT    = "#378ADD",
    ACCENT_H  = "#4A9BE8",
    ACCENT_P  = "#2A6FBB",
    TEXT      = "#E8E8E8",
    TEXT_2    = "#AAAAAA",
    TEXT_3    = "#666666",
    LOG_BG    = "#111111",
    LOG_TEXT  = "#C8C8C8",
    SCROLLH   = "#404040",
    SCROLLHH  = "#555555",
    STATUS_BG = "#378ADD",
    STATUS_FG = "#FFFFFF",
    SEP       = "#333333",
)

# ── Light palette — white / cool grey / blue ──────────────────────────────────
_L = dict(
    BG        = "#F5F5F7",   # Apple-style off-white page
    SURFACE   = "#FFFFFF",   # pure white cards / panels
    SURFACE2  = "#EEEEEF",   # input fields, list backgrounds
    SURFACE3  = "#E3E3E8",   # pressed / deeper inset
    BORDER    = "#DCDCE0",   # default border
    BORDER_S  = "#C8C8CF",   # strong border
    ACCENT    = "#378ADD",   # same blue as dark theme
    ACCENT_H  = "#4A9BE8",
    ACCENT_P  = "#2A6FBB",
    TEXT      = "#1D1D1F",   # near-black
    TEXT_2    = "#6E6E73",   # secondary grey
    TEXT_3    = "#AEAEB2",   # muted / placeholders
    LOG_BG    = "#1A1A1A",   # log stays dark
    LOG_TEXT  = "#C8C8C8",
    SCROLLH   = "#C0C0C8",
    SCROLLHH  = "#A8A8B4",
    STATUS_BG = "#378ADD",   # blue status bar matches dark theme
    STATUS_FG = "#FFFFFF",
    SEP       = "#DCDCE0",
)


def _qss(p: dict) -> str:
    return f"""
/* ── Base ── */
QMainWindow, QDialog {{
    background: {p['BG']};
    color: {p['TEXT']};
}}
QWidget {{
    background: {p['BG']};
    color: {p['TEXT']};
    font-family: "Segoe UI";
    font-size: 10pt;
}}

/* ── Frames used as cards ── */
QFrame[frameShape="4"],
QFrame[frameShape="6"] {{
    background: {p['SURFACE']};
    border: 1px solid {p['BORDER']};
    border-radius: 5px;
}}

/* ── QGroupBox ── */
QGroupBox {{
    background: {p['SURFACE']};
    border: 1px solid {p['BORDER']};
    border-radius: 5px;
    margin-top: 12px;
    padding-top: 4px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {p['TEXT_3']};
}}

/* ── QPushButton — default ── */
QPushButton {{
    background: {p['SURFACE2']};
    color: {p['TEXT']};
    border: 1px solid {p['BORDER_S']};
    border-radius: 4px;
    padding: 4px 12px;
    min-height: 24px;
    font-size: 10pt;
}}
QPushButton:hover   {{ background: {p['SURFACE3']}; border-color: {p['ACCENT']}; }}
QPushButton:pressed {{ background: {p['ACCENT_P']}; color: #fff; border-color: {p['ACCENT_P']}; }}
QPushButton:disabled {{ color: {p['TEXT_3']}; border-color: {p['BORDER']}; background: {p['SURFACE']}; }}

/* Primary — objectName="primaryBtn" */
QPushButton#primaryBtn {{
    background: {p['ACCENT']};
    color: #fff;
    border: none;
    font-weight: 600;
    font-size: 10pt;
    letter-spacing: 0.04em;
}}
QPushButton#primaryBtn:hover   {{ background: {p['ACCENT_H']}; }}
QPushButton#primaryBtn:pressed {{ background: {p['ACCENT_P']}; }}
QPushButton#primaryBtn:disabled {{ background: {p['BORDER']}; color: {p['TEXT_3']}; }}

/* Icon-only toolbar buttons — objectName="iconBtn" */
QPushButton#iconBtn {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 3px;
    min-height: 0;
    min-width: 0;
    color: {p['TEXT_2']};
    font-size: 15px;
}}
QPushButton#iconBtn:hover   {{ background: {p['SURFACE2']}; border-color: {p['BORDER']}; color: {p['TEXT']}; }}
QPushButton#iconBtn:pressed {{ background: {p['SURFACE3']}; }}

/* Plain text link buttons — objectName="linkBtn" */
QPushButton#linkBtn {{
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 1px 0;
    min-height: 0;
    color: {p['TEXT_3']};
    font-size: 9pt;
}}
QPushButton#linkBtn:hover {{ color: {p['ACCENT']}; background: transparent; }}

/* ── QLineEdit ── */
QLineEdit {{
    background: {p['SURFACE2']};
    color: {p['TEXT']};
    border: 1px solid {p['BORDER_S']};
    border-radius: 4px;
    padding: 4px 7px;
    selection-background-color: {p['ACCENT']};
    font-size: 10pt;
}}
QLineEdit:focus     {{ border-color: {p['ACCENT']}; }}
QLineEdit:read-only {{ color: {p['TEXT_2']}; background: {p['SURFACE']}; }}

/* ── QPlainTextEdit — log ── */
QPlainTextEdit {{
    background: {p['LOG_BG']};
    color: {p['LOG_TEXT']};
    border: none;
    border-radius: 0;
    padding: 6px 8px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 9.5pt;
    selection-background-color: {p['ACCENT']};
}}

/* ── QComboBox ── */
QComboBox {{
    background: {p['SURFACE2']};
    color: {p['TEXT']};
    border: 1px solid {p['BORDER_S']};
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
}}
QComboBox:focus {{ border-color: {p['ACCENT']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {p['TEXT_3']};
    width: 0; height: 0; margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {p['SURFACE2']};
    color: {p['TEXT']};
    border: 1px solid {p['BORDER_S']};
    selection-background-color: {p['ACCENT']};
    outline: none;
}}

/* ── QSpinBox / QDoubleSpinBox ── */
QSpinBox, QDoubleSpinBox {{
    background: {p['SURFACE2']};
    color: {p['TEXT']};
    border: 1px solid {p['BORDER_S']};
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 24px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {p['ACCENT']}; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {p['SURFACE3']};
    border: none;
    width: 16px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {p['ACCENT']};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    border-left: 3px solid transparent; border-right: 3px solid transparent;
    border-bottom: 4px solid {p['TEXT_3']}; width: 0; height: 0;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    border-left: 3px solid transparent; border-right: 3px solid transparent;
    border-top: 4px solid {p['TEXT_3']}; width: 0; height: 0;
}}

/* ── QCheckBox ── */
QCheckBox {{
    color: {p['TEXT']};
    spacing: 6px;
    background: transparent;
    font-size: 10pt;
}}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {p['BORDER_S']};
    border-radius: 3px;
    background: {p['SURFACE2']};
}}
QCheckBox::indicator:hover   {{ border-color: {p['ACCENT']}; }}
QCheckBox::indicator:checked {{ background: {p['ACCENT']}; border-color: {p['ACCENT']}; }}

/* ── QSlider ── */
QSlider::groove:horizontal {{
    height: 3px; background: {p['BORDER_S']}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {p['ACCENT']}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {p['ACCENT']}; border: 2px solid {p['BG']};
    width: 13px; height: 13px; margin: -5px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {p['ACCENT_H']}; }}

/* ── QLabel ── */
QLabel {{
    color: {p['TEXT']};
    background: transparent;
}}

/* ── QScrollBar ── */
QScrollBar:vertical {{
    background: transparent; width: 7px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {p['SCROLLH']}; border-radius: 3px; min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {p['SCROLLHH']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 7px;
}}
QScrollBar::handle:horizontal {{
    background: {p['SCROLLH']}; border-radius: 3px; min-width: 20px;
}}
QScrollBar::handle:horizontal:hover {{ background: {p['SCROLLHH']}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── QListWidget (sidebar model list) ── */
QListWidget {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget::item {{
    padding: 4px 8px;
    border-radius: 4px;
    color: {p['TEXT_2']};
    font-size: 9.5pt;
}}
QListWidget::item:hover    {{ background: {p['SURFACE2']}; color: {p['TEXT']}; }}
QListWidget::item:selected {{
    background: rgba(55,138,221,0.12);
    color: {p['ACCENT']};
}}

/* ── QMessageBox ── */
QMessageBox {{ background: {p['SURFACE']}; }}
QMessageBox QLabel {{ background: transparent; }}

/* ── QStatusBar ── */
QStatusBar {{
    background: {p['STATUS_BG']};
    color: {p['STATUS_FG']};
    font-size: 9pt;
}}
QStatusBar::item {{ border: none; }}

/* ── QTabBar (used in dialogs) ── */
QTabWidget::pane {{
    border: 1px solid {p['BORDER']};
    background: {p['BG']};
    border-radius: 5px;
}}
QTabBar::tab {{
    background: {p['SURFACE']};
    color: {p['TEXT_3']};
    border: 1px solid {p['BORDER']};
    border-bottom: none;
    padding: 5px 14px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background: {p['BG']};
    color: {p['TEXT']};
    border-bottom: 2px solid {p['ACCENT']};
}}
QTabBar::tab:hover:!selected {{ color: {p['TEXT_2']}; }}

/* ── QSplitter ── */
QSplitter::handle {{
    background: {p['BORDER']};
}}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical   {{ height: 1px; }}
"""


_CURRENT: str = "dark"
_DARK_QSS  = _qss(_D)
_LIGHT_QSS = _qss(_L)

# Expose palette dicts so other widgets can read colours at runtime
PALETTES = {"dark": _D, "light": _L}


def apply_theme(app: QApplication, theme: str = "dark") -> None:
    global _CURRENT
    _CURRENT = theme
    app.setStyle("Fusion")
    app.setStyleSheet(_DARK_QSS if theme == "dark" else _LIGHT_QSS)


def current_theme() -> str:
    return _CURRENT


def palette() -> dict:
    """Return the active palette dict so widgets can read colours."""
    return PALETTES[_CURRENT]


# ── Title-bar DWM helpers ─────────────────────────────────────────────────────

def _dwm(window, dark: bool) -> None:
    try:
        import ctypes, sys as _sys
        if _sys.platform != "win32":
            return
        hwnd  = int(window.winId())
        val   = ctypes.c_int(1 if dark else 0)
        sz    = ctypes.sizeof(val)
        dwm   = ctypes.windll.dwmapi
        if dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(val), sz) != 0:
            dwm.DwmSetWindowAttribute(hwnd, 19, ctypes.byref(val), sz)
    except Exception:
        pass


def dark_titlebar(w)  -> None: _dwm(w, True)
def light_titlebar(w) -> None: _dwm(w, False)
def auto_titlebar(w)  -> None: _dwm(w, _CURRENT == "dark")
