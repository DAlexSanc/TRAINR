"""
dialogs_extra.py  —  TRAINR
Fully connected replacements for ResumeTrainingDialog and RunComparisonDialog.

Drop this file into Scripts/ and in interface.py replace:

    from dialogs_extra import ResumeTrainingDialog, RunComparisonDialog

Then delete the old class bodies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QProcess, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox,
    QVBoxLayout, QWidget,
)

from theme import auto_titlebar, palette
from ui.tabs.tab_curves import load_results_csv
from paths import YOLO_EXE


# ── colour palette for overlaid runs ─────────────────────────────────────────
_RUN_COLORS = [
    "#378ADD", "#C95F1A", "#2D7A4F", "#BA7517",
    "#7F77DD", "#D4537E", "#639922", "#E24B4A",
]


# ──────────────────────────────────────────────────────────────────────────────
# Resume Training Dialog
# ──────────────────────────────────────────────────────────────────────────────

class ResumeTrainingDialog(QDialog):
    """
    Two modes
    ---------
    True resume  : yolo train resume=True model=last.pt
                   Continues from the exact checkpoint — all hyperparams
                   are read from the checkpoint's args.yaml, nothing else
                   matters.

    Fine-tune    : yolo <task> train model=best.pt data=yaml epochs=N ...
                   Starts fresh epoch counter using existing weights as the
                   starting point.  All Train-tab params apply.
    """

    # Emitted when a run finishes so MainWindow can update status / curves
    run_finished = Signal(int)   # exit code

    def __init__(self, app_state=None, parent=None):
        super().__init__(parent)
        self._parent     = parent   # MainWindow reference for log + status
        self.app_state   = app_state
        self._process: QProcess | None = None

        self.setWindowTitle("Resume / Fine-tune")
        self.setMinimumSize(560, 380)
        self.resize(620, 460)
        self._build()
        auto_titlebar(self)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        # ── Mode selector ─────────────────────────────────────────────────
        mode_frame = QFrame()
        mode_frame.setFrameShape(QFrame.Shape.StyledPanel)
        mode_lay = QHBoxLayout(mode_frame)
        mode_lay.setContentsMargins(10, 8, 10, 8)
        mode_lay.setSpacing(16)

        self._resume_chk = QCheckBox("True resume  (continue from checkpoint, "
                                      "ignores all params below)")
        self._resume_chk.setChecked(False)
        self._resume_chk.toggled.connect(self._on_mode_changed)
        mode_lay.addWidget(self._resume_chk)
        root.addWidget(mode_frame)

        # ── Paths frame ───────────────────────────────────────────────────
        paths_frame = QFrame()
        paths_frame.setFrameShape(QFrame.Shape.StyledPanel)
        g = QGridLayout(paths_frame)
        g.setSpacing(8)
        g.setContentsMargins(10, 10, 10, 10)
        g.setColumnStretch(1, 1)

        def _path_row(row, label, attr, ph, is_file=True, filt="All Files (*.*)"):
            g.addWidget(QLabel(label), row, 0)
            le = QLineEdit()
            le.setPlaceholderText(ph)
            le.setReadOnly(True)
            setattr(self, attr, le)
            g.addWidget(le, row, 1)
            b = QPushButton("Browse")
            b.setFixedWidth(76)
            if is_file:
                b.clicked.connect(
                    lambda _, a=attr, f=filt: self._browse_file(a, f))
            else:
                b.clicked.connect(lambda _, a=attr: self._browse_dir(a))
            g.addWidget(b, row, 2)

        _path_row(0, "Checkpoint (.pt):", "_ckpt_input",
                  "last.pt  or  best.pt",
                  filt="PyTorch weights (*.pt);;All Files (*.*)")
        _path_row(1, "Dataset YAML:", "_yaml_input",
                  "dataset.yaml  (fine-tune only)",
                  filt="YAML (*.yaml *.yml);;All Files (*.*)")
        _path_row(2, "Output folder:", "_out_input",
                  "Where to save the resumed run",
                  is_file=False)

        root.addWidget(paths_frame)

        # ── Fine-tune params ──────────────────────────────────────────────
        self._params_frame = QFrame()
        self._params_frame.setFrameShape(QFrame.Shape.StyledPanel)
        pg = QGridLayout(self._params_frame)
        pg.setSpacing(8)
        pg.setContentsMargins(10, 10, 10, 10)
        pg.setColumnStretch(1, 1)

        pg.addWidget(QLabel("Additional epochs:"), 0, 0)
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 2000)
        self._epochs_spin.setValue(50)
        pg.addWidget(self._epochs_spin, 0, 1)

        pg.addWidget(QLabel("Batch size:"), 1, 0)
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(-1, 1024)
        self._batch_spin.setValue(-1)
        self._batch_spin.setSpecialValueText("Auto (-1)")
        pg.addWidget(self._batch_spin, 1, 1)

        pg.addWidget(QLabel("Image size:"), 2, 0)
        self._imgsz_spin = QSpinBox()
        self._imgsz_spin.setRange(64, 2048)
        self._imgsz_spin.setValue(640)
        self._imgsz_spin.setSingleStep(32)
        pg.addWidget(self._imgsz_spin, 2, 1)

        pg.addWidget(QLabel("Patience:"), 3, 0)
        self._patience_spin = QSpinBox()
        self._patience_spin.setRange(0, 500)
        self._patience_spin.setValue(30)
        pg.addWidget(self._patience_spin, 3, 1)

        hint = QLabel("Regularisation and augmentation params are inherited "
                       "from the Train tab when fine-tuning.")
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"font-size: 8.5pt; color: {palette()['TEXT_3']};")
        pg.addWidget(hint, 4, 0, 1, 2)

        root.addWidget(self._params_frame)

        # ── Log box ───────────────────────────────────────────────────────
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(3000)
        self._log.setPlaceholderText("Run logs will appear here…")
        self._log.setFixedHeight(110)
        root.addWidget(self._log)

        # ── Action buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Start")
        self._run_btn.setObjectName("primaryBtn")
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_mode_changed(self, checked: bool):
        self._params_frame.setEnabled(not checked)
        self._yaml_input.setEnabled(not checked)

    def _browse_file(self, attr: str, filt: str):
        f, _ = QFileDialog.getOpenFileName(self, "Select file", "", filt)
        if f:
            getattr(self, attr).setText(f)

    def _browse_dir(self, attr: str):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            getattr(self, attr).setText(d)

    def _on_run(self):
        ckpt = self._ckpt_input.text().strip()
        if not ckpt:
            QMessageBox.warning(self, "Missing input",
                                "Please select a checkpoint file.")
            return

        if not YOLO_EXE.exists():
            QMessageBox.critical(self, "YOLO not found",
                                 "YOLO CLI not found. Run the Heavy Installer.")
            return

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._log.clear()

        if self._resume_chk.isChecked():
            cmd = [str(YOLO_EXE), "train",
                   f"model={ckpt}",
                   "resume=True"]
            self._log.appendPlainText("True resume mode — continuing from checkpoint…\n")
        else:
            yaml = self._yaml_input.text().strip()
            out  = self._out_input.text().strip()
            if not yaml:
                QMessageBox.warning(self, "Missing input",
                                    "Dataset YAML is required for fine-tuning.")
                self._run_btn.setEnabled(True)
                self._stop_btn.setEnabled(False)
                return

            # Detect task from checkpoint name
            ckpt_name = Path(ckpt).stem.lower()
            task = "segment" if "seg" in ckpt_name else "detect"

            batch = str(self._batch_spin.value())

            cmd = [
                str(YOLO_EXE), task, "train",
                f"model={ckpt}",
                f"data={yaml}",
                f"epochs={self._epochs_spin.value()}",
                f"batch={batch}",
                f"imgsz={self._imgsz_spin.value()}",
                f"patience={self._patience_spin.value()}",
                "exist_ok=True",
            ]
            if out:
                cmd += [f"project={out}", "name=finetune"]
            self._log.appendPlainText(f"Fine-tune mode — task: {task}\n")

        self._log.appendPlainText(" ".join(cmd) + "\n")

        self._process = QProcess(self)
        self._process.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(
            lambda e: self._log.appendPlainText(f"Process error: {e}"))
        self._process.start(cmd[0], cmd[1:])

        # Mirror to main window log if available
        if self._parent and hasattr(self._parent, "log_box"):
            self._parent.log_box.appendPlainText(
                "\n[Resume/Fine-tune] started…")
            if hasattr(self._parent, "status_strip"):
                self._parent.status_strip.set_training(Path(ckpt).name)

    def _read_output(self):
        raw = self._process.readAllStandardOutput().data().decode(errors="ignore")
        if raw:
            self._log.appendPlainText(raw.rstrip())
            if self._parent and hasattr(self._parent, "log_box"):
                self._parent.log_box.appendPlainText(raw.rstrip())
            # Update epoch counter in main status strip
            if self._parent and hasattr(self._parent, "status_strip"):
                for line in raw.splitlines():
                    for token in line.split():
                        if "/" in token:
                            try:
                                c, t = token.split("/", 1)
                                self._parent.status_strip.set_epoch(int(c), int(t))
                                break
                            except ValueError:
                                pass

    def _on_finished(self, exit_code: int, _exit_status):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

        if exit_code == 0:
            self._log.appendPlainText("\n✓ Finished successfully.")
            if self._parent and hasattr(self._parent, "status_strip"):
                self._parent.status_strip.set_done()
        else:
            self._log.appendPlainText(f"\n✗ Process exited with code {exit_code}.")
            if self._parent and hasattr(self._parent, "status_strip"):
                self._parent.status_strip.set_failed("resume/finetune")

        self.run_finished.emit(exit_code)

    def _on_stop(self):
        if self._process and \
                self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
            self._log.appendPlainText("\n— Stopped by user —")


# ──────────────────────────────────────────────────────────────────────────────
# Run Comparison Dialog
# ──────────────────────────────────────────────────────────────────────────────

_COMP_CHART_H = 3.4   # inches per chart
_COMP_CHART_W = 4.8


class _CompChart(FigureCanvas):
    """Single comparison chart — multiple runs overlaid."""

    def __init__(self, parent=None):
        self._fig = Figure(figsize=(_COMP_CHART_W, _COMP_CHART_H),
                           tight_layout=True)
        self._fig.patch.set_facecolor("none")
        super().__init__(self._fig)
        self.setParent(parent)
        self.setFixedHeight(int(_COMP_CHART_H * 96))
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
        ax.grid(color=pal["BORDER"], linewidth=0.5,
                linestyle="--", alpha=0.6)
        return ax

    def plot(self, title: str, runs: list[tuple[str, list]],
             x_label: str = "Epoch",
             y_lim: tuple | None = None):
        ax = self._styled_ax()
        pal = palette()
        for i, (name, vals) in enumerate(runs):
            if vals:
                color = _RUN_COLORS[i % len(_RUN_COLORS)]
                ax.plot(range(1, len(vals) + 1), vals,
                        color=color, linewidth=1.5, label=name)
        ax.set_title(title, fontsize=9, color=pal["TEXT_2"])
        ax.set_xlabel(x_label, fontsize=8)
        if y_lim:
            ax.set_ylim(*y_lim)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=6,
                      framealpha=0.3, labelcolor=pal["TEXT_2"],
                      loc="best")
        self.draw()


class RunComparisonDialog(QDialog):
    def __init__(self, app_state=None, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._runs: list[tuple[str, dict]] = []   # (label, data_dict)
        self.setWindowTitle("Run Comparison")
        self.setMinimumSize(720, 560)
        self.resize(900, 680)
        self._build()
        auto_titlebar(self)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        # ── Add-run frame ─────────────────────────────────────────────────
        top_frame = QFrame()
        top_frame.setFrameShape(QFrame.Shape.StyledPanel)
        top_lay = QVBoxLayout(top_frame)
        top_lay.setSpacing(6)
        top_lay.setContentsMargins(10, 10, 10, 10)

        top_lay.addWidget(QLabel("Add results.csv files to compare:"))

        add_row = QHBoxLayout()
        self._csv_in = QLineEdit()
        self._csv_in.setPlaceholderText("results.csv path")
        self._csv_in.setReadOnly(True)
        add_row.addWidget(self._csv_in, stretch=1)

        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(76)
        browse_btn.clicked.connect(self._browse)
        add_row.addWidget(browse_btn)

        self._label_in = QLineEdit()
        self._label_in.setPlaceholderText("Run label (optional)")
        self._label_in.setFixedWidth(160)
        add_row.addWidget(self._label_in)

        add_btn = QPushButton("Add")
        add_btn.setFixedWidth(56)
        add_btn.clicked.connect(self._add_run)
        add_row.addWidget(add_btn)

        top_lay.addLayout(add_row)

        self._run_list = QPlainTextEdit()
        self._run_list.setReadOnly(True)
        self._run_list.setFixedHeight(64)
        self._run_list.setPlaceholderText("No runs added yet…")
        top_lay.addWidget(self._run_list)

        root.addWidget(top_frame)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        clr_btn = QPushButton("Clear All")
        clr_btn.clicked.connect(self._clear)
        btn_row.addWidget(clr_btn)

        self._compare_btn = QPushButton("Compare")
        self._compare_btn.setObjectName("primaryBtn")
        self._compare_btn.setEnabled(False)
        self._compare_btn.clicked.connect(self._on_compare)
        btn_row.addWidget(self._compare_btn)

        root.addLayout(btn_row)

        # ── Status ────────────────────────────────────────────────────────
        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("font-size: 9pt; color: #888;")
        root.addWidget(self._status)

        # ── Charts in scroll area ─────────────────────────────────────────
        self._charts_widget = QWidget()
        self._charts_widget.setVisible(False)
        charts_lay = QVBoxLayout(self._charts_widget)
        charts_lay.setSpacing(6)
        charts_lay.setContentsMargins(0, 0, 0, 0)

        def _row(*charts):
            w = QWidget()
            l = QHBoxLayout(w)
            l.setSpacing(6)
            l.setContentsMargins(0, 0, 0, 0)
            for c in charts:
                l.addWidget(c)
            return w

        self._c_box   = _CompChart()
        self._c_cls   = _CompChart()
        self._c_map50 = _CompChart()
        self._c_map95 = _CompChart()
        self._c_prec  = _CompChart()
        self._c_rec   = _CompChart()

        charts_lay.addWidget(_row(self._c_box,   self._c_cls))
        charts_lay.addWidget(_row(self._c_map50, self._c_map95))
        charts_lay.addWidget(_row(self._c_prec,  self._c_rec))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._charts_widget)
        root.addWidget(scroll, stretch=1)

        # Placeholder
        self._placeholder = QLabel(
            "Add two or more results.csv files and click Compare")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-size: 12pt;")
        self._placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._placeholder, stretch=1)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select results.csv", "",
            "CSV (*.csv);;All Files (*.*)")
        if f:
            self._csv_in.setText(f)
            # Auto-fill label from parent folder name
            if not self._label_in.text().strip():
                self._label_in.setText(Path(f).parent.name)

    def _add_run(self):
        path = self._csv_in.text().strip()
        if not path:
            return
        if not Path(path).exists():
            QMessageBox.warning(self, "File not found",
                                f"Cannot find:\n{path}")
            return

        label = self._label_in.text().strip() or f"Run {len(self._runs) + 1}"

        # Check for duplicates
        if any(p == path for _, p in [(r[0], r[1].get("_path", ""))
                                       for r in self._runs]):
            QMessageBox.information(self, "Already added",
                                    "This file is already in the list.")
            return

        try:
            data = load_results_csv(Path(path))
            data["_path"] = path   # stash for duplicate check
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        self._runs.append((label, data))
        n = len(data.get("map50", []))
        self._run_list.appendPlainText(
            f"  {len(self._runs)}.  {label}  —  {n} epochs  ({path})")
        self._csv_in.clear()
        self._label_in.clear()
        self._compare_btn.setEnabled(len(self._runs) >= 1)

    def _clear(self):
        self._runs.clear()
        self._run_list.clear()
        self._charts_widget.setVisible(False)
        self._placeholder.setVisible(True)
        self._status.setText("")
        self._compare_btn.setEnabled(False)

    def _on_compare(self):
        if not self._runs:
            return

        self._status.setText(
            f"Comparing {len(self._runs)} run(s)…")
        self._placeholder.setVisible(False)

        def _series(key):
            return [(label, data.get(key, []))
                    for label, data in self._runs]

        self._c_box.plot(  "Box loss",    _series("train_box"))
        self._c_cls.plot(  "Cls loss",    _series("train_cls"))
        self._c_map50.plot("mAP @ 50",    _series("map50"),
                           y_lim=(0, 1.05))
        self._c_map95.plot("mAP @ 50-95", _series("map5095"),
                           y_lim=(0, 1.05))
        self._c_prec.plot( "Precision",   _series("precision"),
                           y_lim=(0, 1.05))
        self._c_rec.plot(  "Recall",      _series("recall"),
                           y_lim=(0, 1.05))

        self._charts_widget.setVisible(True)
        self._status.setText(
            f"{len(self._runs)} run(s) compared  ·  "
            f"click Compare again after adding more runs")
