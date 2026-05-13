"""
theme.py
--------
TRAINR dark theme.

Usage
-----
    from theme import apply_theme
    app = QApplication(sys.argv)
    apply_theme(app)
"""

from PySide6.QtWidgets import QApplication

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#1a1a1a"   # window background
SURFACE   = "#232323"   # frames / cards
SURFACE2  = "#2a2a2a"   # inputs, combos, spinboxes
BORDER    = "#333333"   # subtle borders
ACCENT    = "#378ADD"   # primary blue
ACCENT_H  = "#4a9be8"   # accent hover
ACCENT_P  = "#2a6fbb"   # accent pressed
TEXT      = "#e8e8e8"   # primary text
TEXT_DIM  = "#888888"   # placeholder / labels
DANGER    = "#c0392b"   # destructive actions (future use)
SUCCESS   = "#1d9e75"   # success (future use)

QSS = f"""
/* ── Base ─────────────────────────────────────────────────────────────────── */

QMainWindow, QDialog {{
    background: {BG};
    color: {TEXT};
}}

QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "Segoe UI";
    font-size: 11pt;
}}

/* ── Frames / Cards ───────────────────────────────────────────────────────── */

QFrame[frameShape="4"],
QFrame[frameShape="6"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

/* ── Toolbar ──────────────────────────────────────────────────────────────── */

QToolBar {{
    background: {SURFACE};
    border-bottom: 1px solid {BORDER};
    padding: 4px 6px;
    spacing: 4px;
}}

QToolBar QPushButton {{
    background: {SURFACE2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 12px;
    font-size: 10pt;
}}

QToolBar QPushButton:hover  {{ background: #303030; border-color: {ACCENT}; }}
QToolBar QPushButton:pressed {{ background: {ACCENT_P}; color: #fff; }}

/* ── Buttons ──────────────────────────────────────────────────────────────── */

QPushButton {{
    background: {SURFACE2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 14px;
    min-height: 22px;
}}

QPushButton:hover   {{ background: #303030; border-color: {ACCENT}; color: #fff; }}
QPushButton:pressed {{ background: {ACCENT_P}; border-color: {ACCENT_P}; color: #fff; }}
QPushButton:disabled {{ background: {SURFACE}; color: {TEXT_DIM}; border-color: {BORDER}; }}

/* Primary action button — add property primary="true" or just target by name */
QPushButton#primaryBtn,
QPushButton[text="Start Training"],
QPushButton[text="Organize Dataset"],
QPushButton[text="Analyze Dataset"],
QPushButton[text="Generate Empty Labels"] {{
    background: {ACCENT};
    color: #fff;
    border: none;
    font-weight: 600;
}}

QPushButton#primaryBtn:hover,
QPushButton[text="Start Training"]:hover,
QPushButton[text="Organize Dataset"]:hover,
QPushButton[text="Analyze Dataset"]:hover,
QPushButton[text="Generate Empty Labels"]:hover {{
    background: {ACCENT_H};
}}

QPushButton#primaryBtn:pressed,
QPushButton[text="Start Training"]:pressed,
QPushButton[text="Organize Dataset"]:pressed,
QPushButton[text="Analyze Dataset"]:pressed,
QPushButton[text="Generate Empty Labels"]:pressed {{
    background: {ACCENT_P};
}}

/* ── Inputs ───────────────────────────────────────────────────────────────── */

QLineEdit, QPlainTextEdit, QTextEdit {{
    background: {SURFACE2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: {ACCENT};
}}

QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {ACCENT};
}}

QLineEdit:read-only {{
    color: {TEXT_DIM};
}}

QLineEdit::placeholder {{ color: {TEXT_DIM}; }}

/* Log box — terminal feel */
QPlainTextEdit {{
    font-family: "Consolas", "Courier New", monospace;
    font-size: 10pt;
    background: #111111;
    color: #c8c8c8;
    border-color: #222;
}}

/* ── ComboBox ─────────────────────────────────────────────────────────────── */

QComboBox {{
    background: {SURFACE2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 22px;
}}

QComboBox:focus  {{ border-color: {ACCENT}; }}
QComboBox:hover  {{ border-color: #555; }}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {TEXT_DIM};
    width: 0;
    height: 0;
    margin-right: 6px;
}}

QComboBox QAbstractItemView {{
    background: {SURFACE2};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    outline: none;
}}

/* ── SpinBox ──────────────────────────────────────────────────────────────── */

QSpinBox {{
    background: {SURFACE2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 6px;
    min-height: 22px;
}}

QSpinBox:focus {{ border-color: {ACCENT}; }}

QSpinBox::up-button, QSpinBox::down-button {{
    background: {SURFACE};
    border: none;
    width: 18px;
}}

QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
    background: {ACCENT};
}}

QSpinBox::up-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {TEXT_DIM};
    width: 0; height: 0;
}}

QSpinBox::down-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_DIM};
    width: 0; height: 0;
}}

/* ── CheckBox ─────────────────────────────────────────────────────────────── */

QCheckBox {{
    color: {TEXT};
    spacing: 6px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {SURFACE2};
}}

QCheckBox::indicator:hover   {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: none;
}}

/* Checkmark via border trick */
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

/* ── Slider ───────────────────────────────────────────────────────────────── */

QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {ACCENT};
    border: 2px solid {BG};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    background: {ACCENT_H};
}}

/* ── Labels ───────────────────────────────────────────────────────────────── */

QLabel {{
    color: {TEXT};
    background: transparent;
}}

/* ── ScrollBar ────────────────────────────────────────────────────────────── */

QScrollBar:vertical {{
    background: {BG};
    width: 8px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: #444;
    border-radius: 4px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{ background: #555; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QScrollBar:horizontal {{
    background: {BG};
    height: 8px;
}}

QScrollBar::handle:horizontal {{
    background: #444;
    border-radius: 4px;
    min-width: 24px;
}}

QScrollBar::handle:horizontal:hover {{ background: #555; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── MessageBox ───────────────────────────────────────────────────────────── */

QMessageBox {{
    background: {SURFACE};
}}

QMessageBox QLabel {{ background: transparent; }}

/* ── Tab Widget (future-proof) ────────────────────────────────────────────── */

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: {BG};
}}

QTabBar::tab {{
    background: {SURFACE};
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
    border-bottom: none;
    padding: 6px 16px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}}

QTabBar::tab:selected {{
    background: {BG};
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}

QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
"""


def apply_theme(app: QApplication) -> None:
    """Call once after QApplication is created, before any window is shown."""
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)


def dark_titlebar(window) -> None:
    """
    Tell Windows to render the native title bar in dark mode.
    Works on Windows 10 (build 1809+) and Windows 11.
    Safe no-op on any other platform.

    Call this AFTER window.show():
        window.show()
        dark_titlebar(window)
    """
    try:
        import ctypes
        import sys
        if sys.platform != "win32":
            return

        hwnd = int(window.winId())
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20   # Windows 11 / 10 21H1+
        DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19  # Windows 10 older builds

        value = ctypes.c_int(1)
        size  = ctypes.sizeof(value)

        dwmapi = ctypes.windll.dwmapi
        # Try the modern attribute first, fall back to the older one
        result = dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(value), size,
        )
        if result != 0:
            dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_OLD,
                ctypes.byref(value), size,
            )
    except Exception:
        pass  # never crash the app over a cosmetic call
