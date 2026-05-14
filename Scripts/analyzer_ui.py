from __future__ import annotations

import sys
from pathlib import Path
from theme import dark_titlebar

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from dataset_analyzer import DatasetStats, analyze_dataset

BAR_COLORS = [
    "#378ADD", "#1D9E75", "#D85A30", "#BA7517",
    "#D4537E", "#639922", "#7F77DD", "#E24B4A",
    "#888780", "#0F6E56",
]


# ──────────────────────────────────────────────────────────────────────────────
# Background worker
# ──────────────────────────────────────────────────────────────────────────────

class _AnalyzeWorker(QThread):
    finished = Signal(object)   # DatasetStats
    error    = Signal(str)

    def __init__(self, label_dir: Path, class_names: list[str]):
        super().__init__()
        self._label_dir   = label_dir
        self._class_names = class_names

    def run(self):
        try:
            self.finished.emit(analyze_dataset(self._label_dir, self._class_names))
        except Exception as exc:
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Metric card
# ──────────────────────────────────────────────────────────────────────────────

class _MetricCard(QFrame):
    def __init__(self, label: str, value: str = "—", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("metricCard")
        self.setStyleSheet(
            "#metricCard { background: rgba(128,128,128,0.08);"
            " border-radius: 8px; padding: 4px; }"
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(10, 8, 10, 8)

        self._lbl = QLabel(label)
        self._lbl.setStyleSheet("font-size: 11px; color: #888;")
        self._val = QLabel(value)
        self._val.setStyleSheet("font-size: 16px; font-weight: 600;")

        layout.addWidget(self._lbl)
        layout.addWidget(self._val)

    def set_value(self, v: str):
        self._val.setText(v)


# ──────────────────────────────────────────────────────────────────────────────
# Chart canvas
# ──────────────────────────────────────────────────────────────────────────────

class _ChartCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self._fig = Figure(tight_layout=True)
        self._fig.patch.set_facecolor("none")
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def render(self, stats: DatasetStats):
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        ids = stats.sorted_class_ids()
        if not ids:
            ax.text(0.5, 0.5, "No labeled instances found.",
                    ha="center", va="center", transform=ax.transAxes, color="#888")
            self.draw()
            return

        names  = [stats.class_name(i) for i in ids]
        counts = [stats.class_counts[i] for i in ids]
        total  = stats.total_instances
        colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(ids))]

        y_pos = range(len(ids))
        bars  = ax.barh(y_pos, counts, color=colors, height=0.6, edgecolor="none")

        max_count = max(counts)
        for bar, count in zip(bars, counts):
            pct = 100 * count / total if total else 0
            ax.text(
                bar.get_width() + max_count * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}  ({pct:.1f}%)",
                va="center", ha="left", fontsize=9, color="#aaa",
            )

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Instance count", fontsize=10)
        ax.invert_yaxis()
        ax.set_facecolor("none")
        ax.tick_params(colors="#aaa")
        ax.xaxis.label.set_color("#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.grid(axis="x", color="#333", linewidth=0.5, linestyle="--")
        ax.set_xlim(0, max_count * 1.22)

        self._fig.set_size_inches(7, max(3.5, len(ids) * 0.55 + 1.2))
        self.draw()


# ──────────────────────────────────────────────────────────────────────────────
# Main dialog
# ──────────────────────────────────────────────────────────────────────────────

class DatasetVisualizer(QDialog):
    def __init__(self, app_state=None, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._worker: _AnalyzeWorker | None = None
        self._last_dir: Path | None = None

        self.setWindowTitle("Dataset Visualizer")
        self.setMinimumSize(560, 380)
        self.resize(720, 580)

        self._build_ui()

    # ──────────────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(14, 14, 14, 14)

        # ── input frame ──────────────────────────────────────────────────────
        input_frame = QFrame()
        input_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        grid = QGridLayout(input_frame)
        grid.setSpacing(8)
        grid.setContentsMargins(10, 10, 10, 10)

        # Row 0: labels folder
        grid.addWidget(QLabel("Labels folder:"), 0, 0)
        self._path_input = QLineEdit()
        self._path_input.setPlaceholderText("Path to labels folder (subfolders are scanned automatically)")
        self._path_input.setReadOnly(True)
        grid.addWidget(self._path_input, 0, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        grid.addWidget(browse_btn, 0, 2)

        # Row 1: class names override
        grid.addWidget(QLabel("Class names:"), 1, 0)
        self._class_input = QLineEdit()
        self._class_input.setPlaceholderText(
            "Auto-detected from dataset.yaml or JSON labels — override here if needed"
        )
        grid.addWidget(self._class_input, 1, 1, 1, 2)

        # Row 2: auto-detected names display
        self._detected_label = QLabel("")
        self._detected_label.setStyleSheet("font-size: 11px; color: #888;")
        self._detected_label.setWordWrap(True)
        grid.addWidget(self._detected_label, 2, 1, 1, 2)

        grid.setColumnStretch(1, 1)
        main_layout.addWidget(input_frame)

        # ── analyze button ────────────────────────────────────────────────────
        self._analyze_btn = QPushButton("Analyze Dataset")
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.clicked.connect(self._on_analyze_clicked)
        main_layout.addWidget(self._analyze_btn)

        # ── status label ──────────────────────────────────────────────────────
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 12px; color: #888;")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self._status)

        # ── metric cards ──────────────────────────────────────────────────────
        self._cards_row = QWidget()
        cards_layout = QGridLayout(self._cards_row)
        cards_layout.setSpacing(8)
        cards_layout.setContentsMargins(0, 0, 0, 0)

        self._card_total   = _MetricCard("Total files")
        self._card_labeled = _MetricCard("With labels")
        self._card_empty   = _MetricCard("Empty / bg")
        self._card_inst    = _MetricCard("Total instances")
        self._card_avg     = _MetricCard("Avg instances / img")

        for col, card in enumerate([
            self._card_total, self._card_labeled, self._card_empty,
            self._card_inst, self._card_avg,
        ]):
            cards_layout.addWidget(card, 0, col)

        self._cards_row.setVisible(False)
        main_layout.addWidget(self._cards_row)

        # ── chart in scroll area ──────────────────────────────────────────────
        self._canvas = _ChartCanvas()
        self._canvas.setVisible(False)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidget(self._canvas)
        self._scroll.setVisible(False)
        main_layout.addWidget(self._scroll, stretch=1)

        # ── bottom buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._reload_btn = QPushButton("Re-analyze")
        self._reload_btn.setFixedWidth(100)
        self._reload_btn.setVisible(False)
        self._reload_btn.clicked.connect(self._on_analyze_clicked)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setFixedWidth(90)
        self._reset_btn.setVisible(False)
        self._reset_btn.clicked.connect(self._reset)

        btn_row.addWidget(self._reload_btn)
        btn_row.addWidget(self._reset_btn)
        main_layout.addLayout(btn_row)

    # ──────────────────────────────────────────────────────────────────────────
    # Slots
    # ──────────────────────────────────────────────────────────────────────────

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Labels Folder", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,  # type: ignore
        )
        if not folder:
            return

        self._path_input.setText(folder)
        self._last_dir = Path(folder)
        self._analyze_btn.setEnabled(True)

        # Try to auto-detect class names from yaml and show them as a hint
        from dataset_analyzer import _find_yaml_names
        names = _find_yaml_names(Path(folder))
        if names:
            self._detected_label.setText(f"Auto-detected: {', '.join(names)}")
            # Only pre-fill the override field if it is currently empty
            if not self._class_input.text().strip():
                self._class_input.setText(", ".join(names))
        else:
            self._detected_label.setText(
                "No dataset.yaml found — names will be read from JSON labels or shown as class indices"
            )

    def _on_analyze_clicked(self):
        if not self._last_dir:
            return
        if self._worker and self._worker.isRunning():
            return

        raw = self._class_input.text().strip()
        class_names = [s.strip() for s in raw.split(",") if s.strip()] if raw else []

        self._status.setText(f"Scanning  {self._last_dir}  …")
        self._analyze_btn.setEnabled(False)
        self._cards_row.setVisible(False)
        self._scroll.setVisible(False)
        self._canvas.setVisible(False)
        self._reload_btn.setVisible(False)
        self._reset_btn.setVisible(False)

        self._worker = _AnalyzeWorker(self._last_dir, class_names)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_analysis_done(self, stats: DatasetStats):
        self._status.setText(
            f"Loaded  {stats.label_dir.name}  —  {stats.total_images:,} files"
        )

        # If class names were auto-resolved inside the analyzer, surface them
        if stats.class_names and not self._class_input.text().strip():
            self._class_input.setText(", ".join(stats.class_names))
            self._detected_label.setText(
                f"Auto-detected: {', '.join(stats.class_names)}"
            )

        self._card_total.set_value(f"{stats.total_images:,}")
        self._card_labeled.set_value(
            f"{stats.images_with_labels:,}  ({stats.labeled_pct:.1f}%)"
        )
        self._card_empty.set_value(
            f"{stats.empty_images:,}  ({stats.empty_pct:.1f}%)"
        )
        self._card_inst.set_value(f"{stats.total_instances:,}")
        self._card_avg.set_value(f"{stats.avg_instances_per_labeled_image:.2f}")

        self._cards_row.setVisible(True)
        self._canvas.render(stats)
        self._canvas.setVisible(True)
        self._scroll.setVisible(True)
        self._analyze_btn.setEnabled(True)
        self._reload_btn.setVisible(True)
        self._reset_btn.setVisible(True)

    def _on_analysis_error(self, msg: str):
        self._status.setText("")
        self._analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis error", msg)

    def _reset(self):
        self._last_dir = None
        self._path_input.clear()
        self._class_input.clear()
        self._detected_label.setText("")
        self._status.setText("")
        self._cards_row.setVisible(False)
        self._scroll.setVisible(False)
        self._canvas.setVisible(False)
        self._analyze_btn.setEnabled(False)
        self._reload_btn.setVisible(False)
        self._reset_btn.setVisible(False)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load_directory(self, path: Path | str):
        """Trigger analysis programmatically (e.g. from a project config)."""
        self._path_input.setText(str(path))
        self._last_dir = Path(path)
        self._analyze_btn.setEnabled(True)
        self._on_analyze_clicked()

    def set_class_names(self, names: list[str]):
        """Pre-populate class names override field."""
        self._class_input.setText(", ".join(names))


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    #font = QFont("Segoe UI", 11)
    #app.setFont(font)
    #QApplication.setStyle("Fusion")
    window = DatasetVisualizer()
    window.show()
    dark_titlebar(window)
    sys.exit(app.exec())