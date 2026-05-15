"""
tab_curves.py  —  TRAINR
CurvesTab: reads a YOLO results.csv and renders six same-size training curves
in a vertically scrollable area (2 columns × 3 rows):
  Row 1 : Box loss  |  Cls loss
  Row 2 : mAP@50   |  mAP@50-95
  Row 3 : Precision-Recall curve  |  F1-Confidence curve

The PR and F1 curves are computed from the per-epoch precision/recall values
stored in results.csv.  If the data is too sparse for meaningful curves those
panels show a helpful message.

Emits last_run_ready(map50, map5095, prec, rec, info) after loading.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from theme import palette


# ──────────────────────────────────────────────────────────────────────────────
# CSV parser
# ──────────────────────────────────────────────────────────────────────────────

_WANTED = {
    "epoch":     ["epoch"],
    "train_box": ["train/box_loss", "train/box_loss(b)"],
    "train_cls": ["train/cls_loss"],
    "val_box":   ["val/box_loss",   "val/box_loss(b)"],
    "val_cls":   ["val/cls_loss"],
    "map50":     ["metrics/map50",  "metrics/map50(b)"],
    "map5095":   ["metrics/map50-95", "metrics/map50-95(b)"],
    "precision": ["metrics/precision(b)", "metrics/precision"],
    "recall":    ["metrics/recall(b)",    "metrics/recall"],
}


def _find_col(headers: list[str], candidates: list[str]) -> int | None:
    lower = [h.strip().lower() for h in headers]
    for c in candidates:
        try:
            return lower.index(c.lower())
        except ValueError:
            pass
    return None


def load_results_csv(path: Path) -> dict[str, list]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        col_idx = {k: _find_col(headers, v) for k, v in _WANTED.items()}
        data: dict[str, list] = {k: [] for k in _WANTED}
        for row in reader:
            if not row:
                continue
            for key, idx in col_idx.items():
                if idx is not None and idx < len(row):
                    try:
                        data[key].append(float(row[idx]))
                    except ValueError:
                        pass
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Derived curves helpers
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Background loader
# ──────────────────────────────────────────────────────────────────────────────

class _LoadWorker(QObject):
    finished = Signal(object)
    error    = Signal(str)

    def __init__(self, path: Path):
        super().__init__()
        self._path = path

    def run(self):
        try:
            self.finished.emit(load_results_csv(self._path))
        except Exception as exc:
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Individual chart canvas  (one per subplot so each can be the same fixed size)
# ──────────────────────────────────────────────────────────────────────────────

_CHART_H = 3.8   # inches — controls the height of each row
_CHART_W = 5.0   # inches

class _SingleChart(FigureCanvas):
    def __init__(self, parent=None):
        self._fig = Figure(figsize=(_CHART_W, _CHART_H), tight_layout=True)
        self._fig.patch.set_facecolor("none")
        super().__init__(self._fig)
        self.setParent(parent)
        # Fixed height so all charts are the same size regardless of content
        self.setFixedHeight(int(_CHART_H * 96))   # ~96 dpi
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

    def _styled_ax(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        pal = palette()
        ax.set_facecolor("none")
        ax.tick_params(colors=pal["TEXT_3"], labelsize=7)
        ax.xaxis.label.set_color(pal["TEXT_3"])
        ax.yaxis.label.set_color(pal["TEXT_3"])
        for spine in ax.spines.values():
            spine.set_edgecolor(pal["BORDER"])
        ax.grid(color=pal["BORDER"], linewidth=0.5, linestyle="--", alpha=0.6)
        return ax

    def plot_loss(self, title: str,
                  train_vals: list, val_vals: list,
                  c_train: str, c_val: str):
        ax = self._styled_ax()
        pal = palette()
        if train_vals:
            ax.plot(range(1, len(train_vals) + 1), train_vals,
                    color=c_train, linewidth=1.5,
                    label="train" if val_vals else "")
        if val_vals:
            ax.plot(range(1, len(val_vals) + 1), val_vals,
                    color=c_val, linewidth=1.5, linestyle="--", label="val")
        if val_vals:
            ax.legend(fontsize=7, framealpha=0.3,
                      labelcolor=pal["TEXT_2"])
        ax.set_title(title, fontsize=9, color=pal["TEXT_2"])
        ax.set_xlabel("Epoch", fontsize=8)
        self.draw()

    def plot_metric(self, title: str, vals: list, color: str):
        ax = self._styled_ax()
        pal = palette()
        if vals:
            ax.plot(range(1, len(vals) + 1), vals,
                    color=color, linewidth=1.8)
        ax.set_title(title, fontsize=9, color=pal["TEXT_2"])
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylim(0, 1.05)
        self.draw()

    def plot_from_yolo_png(self, title: str, png_path: Path | None,
                           x_label: str, y_label: str,
                           line_color: str, fill: bool = False,
                           vline_label: str | None = None):
        """
        Extract curve data from a YOLO-generated PNG and replot it in our
        own theme so it matches the other charts visually.

        Strategy
        --------
        YOLO plots on a white background.  Each meaningful curve is drawn in
        a single saturated colour.  We:
          1. Load the PNG as an RGBA array.
          2. Find pixels that are NOT close to white/grey (i.e. curve pixels).
          3. For each pixel column (x) take the median row (y) of coloured
             pixels — this gives a clean single-valued trace even when anti-
             aliasing creates a few stray pixels.
          4. Map pixel coordinates → [0, 1] data coordinates using the
             plot-area bounding box estimated from the white border.
          5. Replot on our styled axes.
        """
        ax = self._styled_ax()
        pal = palette()

        if png_path is None or not png_path.exists():
            ax.text(0.5, 0.5, f"{title}\nnot found in output folder",
                    transform=ax.transAxes, ha="center", va="center",
                    color=pal["TEXT_3"], fontsize=9)
            ax.set_title(title, fontsize=9, color=pal["TEXT_2"])
            self.draw()
            return

        try:
            img_rgba = np.array(
                Image.open(str(png_path)).convert("RGBA"),
                dtype=np.float32
            ) / 255.0

            h_px, w_px = img_rgba.shape[:2]
            r, g, b = img_rgba[..., 0], img_rgba[..., 1], img_rgba[..., 2]

            # ── locate plot area by finding the first/last non-white row/col ──
            # "White" = all channels > 0.92
            white = (r > 0.92) & (g > 0.92) & (b > 0.92)
            non_white_rows = np.where(~white.all(axis=1))[0]
            non_white_cols = np.where(~white.all(axis=0))[0]

            if len(non_white_rows) < 10 or len(non_white_cols) < 10:
                raise ValueError("Could not locate plot area in PNG.")

            row_min, row_max = int(non_white_rows[0]),  int(non_white_rows[-1])
            col_min, col_max = int(non_white_cols[0]),  int(non_white_cols[-1])

            # Crop to the plot region
            crop_r = img_rgba[row_min:row_max, col_min:col_max]
            ch, cw = crop_r.shape[:2]
            cr, cg, cb = crop_r[..., 0], crop_r[..., 1], crop_r[..., 2]

            # ── isolate coloured (non-grey) pixels ────────────────────────────
            # Saturation proxy: max-channel − min-channel
            sat = (np.maximum(np.maximum(cr, cg), cb) -
                   np.minimum(np.minimum(cr, cg), cb))
            # Also exclude very dark pixels (axes ticks, text)
            bright = (cr + cg + cb) / 3 > 0.25
            coloured = (sat > 0.25) & bright   # shape (ch, cw)

            xs_data, ys_data = [], []
            for col in range(cw):
                rows_in_col = np.where(coloured[:, col])[0]
                if len(rows_in_col) == 0:
                    continue
                row_med = int(np.median(rows_in_col))
                # Map to [0, 1]: x left→right, y bottom→top
                x_norm = col  / max(cw - 1, 1)
                y_norm = 1.0 - row_med / max(ch - 1, 1)
                xs_data.append(x_norm)
                ys_data.append(y_norm)

            if len(xs_data) < 5:
                raise ValueError("Too few curve pixels extracted.")

            # Sort by x in case any columns were skipped
            pairs = sorted(zip(xs_data, ys_data))
            xs_data = [p[0] for p in pairs]
            ys_data = [p[1] for p in pairs]

            ax.plot(xs_data, ys_data, color=line_color, linewidth=1.8)

            if fill:
                ax.fill_between(xs_data, ys_data, alpha=0.10, color=line_color)

            # Best-point annotation (max y)
            if vline_label and ys_data:
                best_idx = int(np.argmax(ys_data))
                bx, by   = xs_data[best_idx], ys_data[best_idx]
                ax.axvline(bx, color=line_color, linestyle=":",
                           linewidth=1, alpha=0.55)
                offset = 0.03 if bx < 0.85 else -0.03
                ha     = "left" if bx < 0.85 else "right"
                ax.text(bx + offset, 0.06,
                        f"{vline_label} {by:.3f}\n@ {bx:.2f}",
                        fontsize=7, color=pal["TEXT_2"], ha=ha)

        except Exception as e:
            ax.text(0.5, 0.5, f"Could not parse image\n{e}",
                    transform=ax.transAxes, ha="center", va="center",
                    color=pal["TEXT_3"], fontsize=8)

        ax.set_title(title, fontsize=9, color=pal["TEXT_2"])
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.08)
        self.draw()


# ──────────────────────────────────────────────────────────────────────────────
# CurvesTab
# ──────────────────────────────────────────────────────────────────────────────

class CurvesTab(QWidget):
    last_run_ready = Signal(float, float, float, float, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _LoadWorker | None = None
        self._current_path: str = ""
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        # Input row
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        row = QHBoxLayout(frame)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)

        row.addWidget(QLabel("results.csv:"))
        self._csv_in = QLineEdit()
        self._csv_in.setPlaceholderText(
            "Select a YOLO results.csv  —  or finish a training run")
        self._csv_in.setReadOnly(True)
        row.addWidget(self._csv_in, stretch=1)

        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(76)
        browse_btn.clicked.connect(self._browse)
        row.addWidget(browse_btn)

        self._load_btn = QPushButton("Load")
        self._load_btn.setObjectName("primaryBtn")
        self._load_btn.setFixedWidth(64)
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._on_load)
        row.addWidget(self._load_btn)

        lay.addWidget(frame)

        # Status
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 9pt; color: #888;")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._status)

        # Scrollable chart grid
        self._charts_widget = QWidget()
        self._charts_widget.setVisible(False)
        charts_lay = QVBoxLayout(self._charts_widget)
        charts_lay.setSpacing(6)
        charts_lay.setContentsMargins(0, 0, 0, 0)

        # Three rows, two charts each
        def _chart_row(*charts):
            row_w = QWidget()
            row_lay = QHBoxLayout(row_w)
            row_lay.setSpacing(6)
            row_lay.setContentsMargins(0, 0, 0, 0)
            for c in charts:
                row_lay.addWidget(c)
            return row_w

        self._c_box_loss  = _SingleChart()
        self._c_cls_loss  = _SingleChart()
        self._c_map50     = _SingleChart()
        self._c_map5095   = _SingleChart()
        self._c_pr        = _SingleChart()
        self._c_f1        = _SingleChart()

        charts_lay.addWidget(_chart_row(self._c_box_loss, self._c_cls_loss))
        charts_lay.addWidget(_chart_row(self._c_map50,    self._c_map5095))
        charts_lay.addWidget(_chart_row(self._c_pr,       self._c_f1))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._charts_widget)
        self._scroll = scroll
        lay.addWidget(scroll, stretch=1)

        # Placeholder
        self._placeholder = QLabel(
            "Load a results.csv to display training curves")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-size: 12pt;")
        self._placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self._placeholder, stretch=1)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select results.csv", "",
            "CSV (*.csv);;All Files (*.*)")
        if f:
            self._csv_in.setText(f)
            self._current_path = f
            self._load_btn.setEnabled(True)

    def _on_load(self):
        path = Path(self._csv_in.text().strip())
        if not path.exists():
            QMessageBox.warning(self, "File not found", f"Cannot find:\n{path}")
            return
        self._run_load(path)

    def _run_load(self, path: Path):
        if self._thread and self._thread.isRunning():
            return

        self._load_btn.setEnabled(False)
        self._status.setText(f"Loading  {path.name} …")
        self._placeholder.setVisible(False)
        self._charts_widget.setVisible(False)

        self._thread = QThread()
        self._worker = _LoadWorker(path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_data)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_data(self, data: dict):
        self._load_btn.setEnabled(True)
        n = len(data.get("epoch", data.get("map50", [])))
        self._status.setText(
            f"{n} epochs  ·  {Path(self._csv_in.text()).name}")

        # Loss curves
        self._c_box_loss.plot_loss(
            "Box loss",
            data["train_box"], data["val_box"],
            palette()["ACCENT"], "#378ADD")
        self._c_cls_loss.plot_loss(
            "Cls loss",
            data["train_cls"], data["val_cls"],
            "#D85A30", "#7F77DD")

        # mAP curves
        self._c_map50.plot_metric(  "mAP @ 50",    data["map50"],   "#2D7A4F")
        self._c_map5095.plot_metric("mAP @ 50-95", data["map5095"], "#BA7517")

        # PR and F1 — extract from YOLO's own PNGs and replot in our theme
        train_dir = Path(self._csv_in.text()).parent
        pr_png  = next(iter(train_dir.glob("*PR_curve.png")),  None)
        f1_png  = next(iter(train_dir.glob("*F1_curve.png")),  None)

        self._c_pr.plot_from_yolo_png(
            "Precision-Recall", pr_png,
            x_label="Recall", y_label="Precision",
            line_color="#2D7A4F", fill=True)
        self._c_f1.plot_from_yolo_png(
            "F1-Confidence", f1_png,
            x_label="Confidence", y_label="F1",
            line_color=palette()["ACCENT"],
            vline_label="best")

        self._charts_widget.setVisible(True)

        # Feed Last Run card
        def _last(key):
            vals = data.get(key, [])
            return vals[-1] if vals else 0.0

        self.last_run_ready.emit(
            _last("map50"), _last("map5095"),
            _last("precision"), _last("recall"),
            f"epoch {n}  ·  {Path(self._csv_in.text()).stem}",
        )

    def _on_error(self, msg: str):
        self._load_btn.setEnabled(True)
        self._status.setText("")
        self._placeholder.setVisible(True)
        QMessageBox.critical(self, "Load error", msg)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_csv(self, path: str):
        self._csv_in.setText(path)
        self._current_path = path
        self._load_btn.setEnabled(True)
        self._run_load(Path(path))
