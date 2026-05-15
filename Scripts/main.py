"""
main.py  —  TRAINR entry point
Run: python main.py
"""
import sys
from PySide6.QtWidgets import QApplication
from core.app_state import AppState
from ui.main_window import MainWindow
from theme import apply_theme, auto_titlebar
from paths import CONFIG

app   = QApplication(sys.argv)
state = AppState(str(CONFIG))

apply_theme(app, state.get("ui.theme", "dark"))

window = MainWindow(state)
window.show()
auto_titlebar(window)

app.exec()
