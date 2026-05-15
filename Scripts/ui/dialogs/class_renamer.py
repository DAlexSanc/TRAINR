"""
class_renamer.py  —  TRAINR
Rename and reorder YOLO classes.

Operations
----------
1. Load dataset.yaml → parse names block
2. User edits names and drags rows to reorder
3. On confirm:
   - Rewrite every .txt in labels_dir remapping old → new class indices
   - Rewrite dataset.yaml with new names in new order

Standalone:  python class_renamer.py
From main:   ClassRenamerDialog(parent=self).exec()
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QHeaderView, QAbstractItemView,
)

from theme import auto_titlebar, apply_theme, palette


# ──────────────────────────────────────────────────────────────────────────────
# Backend — pure Python, no GUI dependency
# ──────────────────────────────────────────────────────────────────────────────

def _parse_yaml_names(text: str) -> list[str]:
    """Minimal parser for the names: block — no pyyaml dependency."""
    lines = text.splitlines()
    in_names = False
    names: dict[int, str] = {}

    for line in lines:
        s = line.strip()
        if s.lower().startswith("names:") and "[" in s:
            inner = s.split("[", 1)[1].split("]")[0]
            return [n.strip().strip("'\"") for n in inner.split(",") if n.strip()]
        if s.lower().startswith("names:"):
            in_names = True
            continue
        if in_names:
            if s == "" or (not line.startswith(" ") and not line.startswith("\t")):
                in_names = False
                continue
            if ":" in s:
                idx_s, _, name = s.partition(":")
                try:
                    names[int(idx_s.strip())] = name.strip().strip("'\"")
                except ValueError:
                    pass
            elif s.startswith("-"):
                names[len(names)] = s.lstrip("- ").strip().strip("'\"")

    return [names[i] for i in sorted(names)] if names else []


def _rewrite_yaml(yaml_path: Path, new_names: list[str]) -> None:
    """Rewrite the names block in dataset.yaml, preserving all other keys."""
    text   = yaml_path.read_text(encoding="utf-8", errors="ignore")
    lines  = text.splitlines()
    output = []
    skip   = False

    for line in lines:
        s = line.strip()
        if s.lower().startswith("names:"):
            skip = True
            output.append("names:")
            for i, name in enumerate(new_names):
                output.append(f"  {i}: {name}")
            continue
        if skip:
            # Keep skipping until we hit a non-indented, non-empty line
            if s == "" or (not line.startswith(" ") and not line.startswith("\t")):
                skip = False
                output.append(line)
            # else: drop (old names block)
        else:
            output.append(line)

    yaml_path.write_text("\n".join(output), encoding="utf-8")


def remap_labels(
    labels_dir: Path,
    old_to_new: dict[int, int],
    new_names: list[str],
    yaml_path: Path | None = None,
) -> dict:
    """
    Rewrite every YOLO .txt file in labels_dir (recursively) replacing
    old class indices with new ones.  Lines whose class id is not in
    old_to_new are dropped (class was deleted).

    Parameters
    ----------
    labels_dir : folder containing *.txt label files (scanned recursively)
    old_to_new : mapping  old_index → new_index
    new_names  : ordered list of new class names (for yaml rewrite)
    yaml_path  : if provided, dataset.yaml is rewritten too

    Returns
    -------
    dict  { "files_updated": int, "lines_remapped": int, "lines_dropped": int }
    """
    files_updated  = 0
    lines_remapped = 0
    lines_dropped  = 0

    for txt in sorted(labels_dir.rglob("*.txt")):
        original = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        new_lines = []
        changed   = False

        for line in original:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                old_cls = int(parts[0])
            except ValueError:
                new_lines.append(line)   # keep malformed lines unchanged
                continue

            if old_cls in old_to_new:
                new_cls = old_to_new[old_cls]
                if new_cls != old_cls:
                    parts[0] = str(new_cls)
                    changed = True
                    lines_remapped += 1
                new_lines.append(" ".join(parts))
            else:
                # Class was deleted — drop the line
                lines_dropped += 1
                changed = True

        if changed:
            txt.write_text("\n".join(new_lines), encoding="utf-8")
            files_updated += 1

    if yaml_path and yaml_path.exists():
        _rewrite_yaml(yaml_path, new_names)

    return {
        "files_updated":  files_updated,
        "lines_remapped": lines_remapped,
        "lines_dropped":  lines_dropped,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────────

class _Worker(QObject):
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, labels_dir: Path, old_to_new: dict,
                 new_names: list[str], yaml_path: Path | None):
        super().__init__()
        self._labels_dir = labels_dir
        self._old_to_new = old_to_new
        self._new_names  = new_names
        self._yaml_path  = yaml_path

    def run(self):
        try:
            result = remap_labels(
                self._labels_dir, self._old_to_new,
                self._new_names, self._yaml_path,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Class table widget  (reorderable + editable)
# ──────────────────────────────────────────────────────────────────────────────

class _ClassTable(QTableWidget):
    """
    Two-column table: original index (read-only) | class name (editable).
    Rows can be dragged to reorder.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["#", "Class name"])
        self.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setAlternatingRowColors(True)

    def load(self, names: list[str]):
        self.setRowCount(0)
        for i, name in enumerate(names):
            self.insertRow(i)
            idx_item = QTableWidgetItem(str(i))
            idx_item.setFlags(Qt.ItemFlag.ItemIsEnabled)   # not editable
            idx_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            self.setItem(i, 0, idx_item)
            self.setItem(i, 1, QTableWidgetItem(name))

    def current_names(self) -> list[str]:
        """Return class names in current visual order."""
        return [
            self.item(r, 1).text().strip()
            for r in range(self.rowCount())
            if self.item(r, 1)
        ]

    def original_indices(self) -> list[int]:
        """
        Return the original class index for each row in current visual order.
        Used to build the old→new remap.
        """
        result = []
        for r in range(self.rowCount()):
            item = self.item(r, 0)
            if item:
                try:
                    result.append(int(item.text()))
                except ValueError:
                    result.append(r)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────

class ClassRenamerDialog(QDialog):
    def __init__(self, app_state=None, parent=None):
        super().__init__(parent)
        self.app_state   = app_state
        self._yaml_path: Path | None  = None
        self._labels_dir: Path | None = None
        self._thread = None
        self._worker = None

        self.setWindowTitle("Class Renamer & Reorder")
        self.setMinimumSize(500, 420)
        self.resize(560, 520)
        self._build()
        auto_titlebar(self)

    # ──────────────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        # ── Input frame ───────────────────────────────────────────────────
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        g = QGridLayout(frame)
        g.setSpacing(8)
        g.setContentsMargins(10, 10, 10, 10)
        g.setColumnStretch(1, 1)

        # Dataset YAML
        g.addWidget(QLabel("Dataset YAML:"), 0, 0)
        self._yaml_input = QLineEdit()
        self._yaml_input.setPlaceholderText("dataset.yaml  —  class names are read from here")
        self._yaml_input.setReadOnly(True)
        g.addWidget(self._yaml_input, 0, 1)
        b1 = QPushButton("Browse")
        b1.setFixedWidth(76)
        b1.clicked.connect(self._browse_yaml)
        g.addWidget(b1, 0, 2)

        # Labels folder
        g.addWidget(QLabel("Labels folder:"), 1, 0)
        self._labels_input = QLineEdit()
        self._labels_input.setPlaceholderText("Folder containing *.txt label files")
        self._labels_input.setReadOnly(True)
        g.addWidget(self._labels_input, 1, 1)
        b2 = QPushButton("Browse")
        b2.setFixedWidth(76)
        b2.clicked.connect(self._browse_labels)
        g.addWidget(b2, 1, 2)

        root.addWidget(frame)

        # ── Hint label ────────────────────────────────────────────────────
        hint = QLabel(
            "Drag rows to reorder  ·  double-click a name to rename  ·  "
            "deleted rows will have their labels removed from all .txt files"
        )
        hint.setStyleSheet(f"font-size: 8.5pt; color: {palette()['TEXT_3']};")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # ── Class table ───────────────────────────────────────────────────
        self._table = _ClassTable()
        root.addWidget(self._table, stretch=1)

        # ── Row action buttons ────────────────────────────────────────────
        row_btns = QHBoxLayout()
        row_btns.setSpacing(6)

        self._del_btn = QPushButton("Delete selected")
        self._del_btn.clicked.connect(self._delete_selected)
        row_btns.addWidget(self._del_btn)
        row_btns.addStretch()

        root.addLayout(row_btns)

        # ── Apply button ──────────────────────────────────────────────────
        self._apply_btn = QPushButton("Apply Changes")
        self._apply_btn.setObjectName("primaryBtn")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        root.addWidget(self._apply_btn)

    # ──────────────────────────────────────────────────────────────────────────
    # Slots
    # ──────────────────────────────────────────────────────────────────────────

    def _browse_yaml(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select dataset.yaml", "",
            "YAML (*.yaml *.yml);;All Files (*.*)")
        if not f:
            return
        self._yaml_path = Path(f)
        self._yaml_input.setText(f)
        self._load_classes_from_yaml(self._yaml_path)
        self._update_apply_btn()

    def _browse_labels(self):
        d = QFileDialog.getExistingDirectory(self, "Select labels folder")
        if not d:
            return
        self._labels_dir = Path(d)
        self._labels_input.setText(d)
        self._update_apply_btn()

    def _load_classes_from_yaml(self, path: Path):
        try:
            text  = path.read_text(encoding="utf-8", errors="ignore")
            names = _parse_yaml_names(text)
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Could not read YAML:\n{e}")
            return

        if not names:
            QMessageBox.warning(self, "No classes found",
                                "No names: block found in the selected YAML.")
            return

        self._table.load(names)

    def _delete_selected(self):
        rows = sorted(
            {idx.row() for idx in self._table.selectedIndexes()},
            reverse=True,
        )
        for r in rows:
            self._table.removeRow(r)

    def _update_apply_btn(self):
        self._apply_btn.setEnabled(
            self._yaml_path is not None and self._labels_dir is not None
        )

    def _on_apply(self):
        if self._table.rowCount() == 0:
            QMessageBox.warning(self, "Nothing to do", "The class list is empty.")
            return

        new_names      = self._table.current_names()
        original_order = self._table.original_indices()

        # Validate — no blank names
        if any(n == "" for n in new_names):
            QMessageBox.warning(self, "Empty name",
                                "One or more class names are blank. Please fill them in.")
            return

        # Build old_index → new_index map from the drag reorder
        old_to_new = {old: new for new, old in enumerate(original_order)}

        # Confirmation
        lines = [f"  {old} → {new}  \"{new_names[new]}\""
                 for old, new in sorted(old_to_new.items())]
        msg = (
            f"This will rewrite all .txt files in:\n  {self._labels_dir}\n\n"
            f"Class remapping:\n" + "\n".join(lines) +
            "\n\nThis cannot be undone. Continue?"
        )
        if QMessageBox.question(
            self, "Confirm", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return

        self._apply_btn.setEnabled(False)
        self._apply_btn.setText("Working…")

        from PySide6.QtCore import QThread
        self._thread = QThread()
        self._worker = _Worker(
            self._labels_dir, old_to_new, new_names,
            self._yaml_path,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_done(self, result: dict):
        self._apply_btn.setEnabled(True)
        self._apply_btn.setText("Apply Changes")
        QMessageBox.information(
            self, "Done",
            f"Changes applied successfully.\n\n"
            f"Files updated:   {result['files_updated']}\n"
            f"Lines remapped:  {result['lines_remapped']}\n"
            f"Lines dropped:   {result['lines_dropped']}  (deleted classes)",
        )

    def _on_error(self, msg: str):
        self._apply_btn.setEnabled(True)
        self._apply_btn.setText("Apply Changes")
        QMessageBox.critical(self, "Error", msg)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_theme(app, "dark")
    window = ClassRenamerDialog()
    window.show()
    auto_titlebar(window)
    sys.exit(app.exec())
