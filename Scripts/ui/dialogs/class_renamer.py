"""
class_renamer.py  —  TRAINR
Rename and reorder YOLO classes.

Fixes vs previous version
--------------------------
- Custom dropEvent prevents rows from being deleted on drag-reorder.
- Original indices tracked in a separate list (not read from column 0),
  so the old→new map is always accurate regardless of how many times
  rows are dragged.
- Worker emits progress so the UI shows a running file count.
- Confirmation dialog shows a full diff preview before touching anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QHeaderView, QAbstractItemView,
    QProgressBar,
)

from theme import auto_titlebar, apply_theme, palette


# ──────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_yaml_names(text: str) -> list[str]:
    lines    = text.splitlines()
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
            if s == "" or (not line.startswith(" ") and not line.startswith("\t")):
                skip = False
                output.append(line)
        else:
            output.append(line)

    yaml_path.write_text("\n".join(output), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Worker  (emits progress per file)
# ──────────────────────────────────────────────────────────────────────────────

class _Worker(QObject):
    progress = Signal(int, int)   # (files_done, files_total)
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, labels_dir: Path, old_to_new: dict[int, int],
                 new_names: list[str], yaml_path: Path | None,
                 deleted_ids: set[int]):
        super().__init__()
        self._labels_dir = labels_dir
        self._old_to_new = old_to_new
        self._new_names  = new_names
        self._yaml_path  = yaml_path
        self._deleted    = deleted_ids

    def run(self):
        try:
            txt_files = sorted(self._labels_dir.rglob("*.txt"))
            total     = len(txt_files)

            files_updated  = 0
            lines_remapped = 0
            lines_dropped  = 0

            for done, txt in enumerate(txt_files, 1):
                self.progress.emit(done, total)

                original  = txt.read_text(
                    encoding="utf-8", errors="ignore").splitlines()
                new_lines = []
                changed   = False

                for line in original:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        old_cls = int(parts[0])
                    except ValueError:
                        new_lines.append(line)
                        continue

                    if old_cls in self._deleted:
                        # Class was explicitly deleted — drop the annotation
                        lines_dropped += 1
                        changed = True
                        continue

                    new_cls = self._old_to_new.get(old_cls, old_cls)
                    if new_cls != old_cls:
                        parts[0] = str(new_cls)
                        changed  = True
                        lines_remapped += 1
                    new_lines.append(" ".join(parts))

                if changed:
                    txt.write_text("\n".join(new_lines), encoding="utf-8")
                    files_updated += 1

            if self._yaml_path and self._yaml_path.exists():
                _rewrite_yaml(self._yaml_path, self._new_names)

            self.finished.emit({
                "files_scanned":  total,
                "files_updated":  files_updated,
                "lines_remapped": lines_remapped,
                "lines_dropped":  lines_dropped,
            })

        except Exception as exc:
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Class table  —  custom drag-reorder that never deletes rows
# ──────────────────────────────────────────────────────────────────────────────

class _ClassTable(QTableWidget):
    """
    Two-column table:
      col 0 — original index  (display only, reflects load-time position)
      col 1 — class name      (editable via double-click)

    Reorder using the ↑↓ buttons — drag-drop is disabled to prevent
    Qt's InternalMove handler from silently deleting rows.

    original_order() returns the load-time index for each current row so
    the caller can build the correct old→new remap.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Original #", "Class name"])
        self.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.setAlternatingRowColors(True)

        # Parallel list tracking the original (load-time) index per row.
        # Updated whenever rows are swapped.
        self._orig_indices: list[int] = []

    # ── load ──────────────────────────────────────────────────────────────────

    def load(self, names: list[str]):
        self.setRowCount(0)
        self._orig_indices = list(range(len(names)))
        for i, name in enumerate(names):
            self._insert_row(i, i, name)

    def _insert_row(self, visual_row: int, orig_idx: int, name: str):
        self.insertRow(visual_row)
        idx_item = QTableWidgetItem(str(orig_idx))
        idx_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        idx_item.setTextAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.setItem(visual_row, 0, idx_item)
        self.setItem(visual_row, 1, QTableWidgetItem(name))

    # ── drag-reorder disabled — use ↑↓ buttons instead ───────────────────────

    def _swap_rows(self, a: int, b: int):
        # Swap column 0 text (original index label)
        t0a = self.item(a, 0).text()
        t0b = self.item(b, 0).text()
        self.item(a, 0).setText(t0b)
        self.item(b, 0).setText(t0a)

        # Swap column 1 text (class name)
        t1a = self.item(a, 1).text()
        t1b = self.item(b, 1).text()
        self.item(a, 1).setText(t1b)
        self.item(b, 1).setText(t1a)

        # Swap the parallel tracking list
        self._orig_indices[a], self._orig_indices[b] = \
            self._orig_indices[b], self._orig_indices[a]

    # ── public API ────────────────────────────────────────────────────────────

    def current_names(self) -> list[str]:
        return [
            self.item(r, 1).text().strip()
            for r in range(self.rowCount())
            if self.item(r, 1)
        ]

    def original_order(self) -> list[int]:
        """
        Returns the original (load-time) index for each row in current
        visual order.  E.g. if the user moved class 2 to position 0:
            [2, 0, 1, ...]
        """
        return list(self._orig_indices)

    def remove_selected(self) -> list[int]:
        """
        Remove currently selected rows.
        Returns the original indices that were removed (for the deleted set).
        """
        rows = sorted(
            {idx.row() for idx in self.selectedIndexes()},
            reverse=True,
        )
        removed_orig = []
        for r in rows:
            removed_orig.append(self._orig_indices[r])
            self._orig_indices.pop(r)
            self.removeRow(r)
        return removed_orig


# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────

class ClassRenamerDialog(QDialog):
    def __init__(self, app_state=None, parent=None):
        super().__init__(parent)
        self.app_state    = app_state
        self._yaml_path: Path | None  = None
        self._labels_dir: Path | None = None
        self._deleted_orig: set[int]  = set()   # original indices of deleted classes
        self._thread: QThread | None  = None
        self._worker: _Worker | None  = None

        self.setWindowTitle("Class Renamer & Reorder")
        self.setMinimumSize(520, 460)
        self.resize(580, 560)
        self._build()
        auto_titlebar(self)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        # Input frame
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        g = QGridLayout(frame)
        g.setSpacing(8)
        g.setContentsMargins(10, 10, 10, 10)
        g.setColumnStretch(1, 1)

        g.addWidget(QLabel("Dataset YAML:"), 0, 0)
        self._yaml_input = QLineEdit()
        self._yaml_input.setPlaceholderText("dataset.yaml  —  class names are read from here")
        self._yaml_input.setReadOnly(True)
        g.addWidget(self._yaml_input, 0, 1)
        b1 = QPushButton("Browse")
        b1.setFixedWidth(76)
        b1.clicked.connect(self._browse_yaml)
        g.addWidget(b1, 0, 2)

        g.addWidget(QLabel("Labels folder:"), 1, 0)
        self._labels_input = QLineEdit()
        self._labels_input.setPlaceholderText("Folder containing *.txt label files (scanned recursively)")
        self._labels_input.setReadOnly(True)
        g.addWidget(self._labels_input, 1, 1)
        b2 = QPushButton("Browse")
        b2.setFixedWidth(76)
        b2.clicked.connect(self._browse_labels)
        g.addWidget(b2, 1, 2)

        root.addWidget(frame)

        # Hint
        hint = QLabel(
            "Drag rows to reorder  ·  double-click a name to rename  ·  "
            "delete a row to remove that class from all label files")
        hint.setStyleSheet(f"font-size: 8.5pt; color: {palette()['TEXT_3']};")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # Table
        self._table = _ClassTable()
        root.addWidget(self._table, stretch=1)

        # Row buttons
        row_btns = QHBoxLayout()
        self._del_btn = QPushButton("Delete selected class")
        self._del_btn.clicked.connect(self._delete_selected)
        row_btns.addWidget(self._del_btn)
        row_btns.addStretch()

        # Move up / down buttons as an alternative to dragging
        up_btn = QPushButton("↑")
        up_btn.setFixedWidth(34)
        up_btn.setToolTip("Move selected row up")
        up_btn.clicked.connect(self._move_up)
        dn_btn = QPushButton("↓")
        dn_btn.setFixedWidth(34)
        dn_btn.setToolTip("Move selected row down")
        dn_btn.clicked.connect(self._move_down)
        row_btns.addWidget(up_btn)
        row_btns.addWidget(dn_btn)
        root.addLayout(row_btns)

        # Progress bar (hidden until apply is clicked)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        self._progress.setFormat("Processing file %v of %m…")
        root.addWidget(self._progress)

        # Status label
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 9pt; color: #888;")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._status)

        # Apply button
        self._apply_btn = QPushButton("Apply Changes")
        self._apply_btn.setObjectName("primaryBtn")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        root.addWidget(self._apply_btn)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _browse_yaml(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select dataset.yaml", "",
            "YAML (*.yaml *.yml);;All Files (*.*)")
        if not f:
            return
        self._yaml_path = Path(f)
        self._yaml_input.setText(f)
        self._deleted_orig.clear()
        self._load_from_yaml(self._yaml_path)
        self._update_apply_btn()

    def _browse_labels(self):
        d = QFileDialog.getExistingDirectory(self, "Select labels folder")
        if not d:
            return
        self._labels_dir = Path(d)
        self._labels_input.setText(d)
        self._update_apply_btn()

    def _load_from_yaml(self, path: Path):
        try:
            text  = path.read_text(encoding="utf-8", errors="ignore")
            names = _parse_yaml_names(text)
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Could not read YAML:\n{e}")
            return
        if not names:
            QMessageBox.warning(self, "No classes found",
                                "No names: block found in the YAML.")
            return
        self._table.load(names)
        self._status.setText(f"Loaded {len(names)} classes from YAML.")

    def _delete_selected(self):
        removed = self._table.remove_selected()
        self._deleted_orig.update(removed)
        if removed:
            self._status.setText(
                f"Marked {len(self._deleted_orig)} class(es) for deletion.")

    def _move_up(self):
        r = self._table.currentRow()
        if r > 0:
            self._table._swap_rows(r, r - 1)
            self._table.selectRow(r - 1)

    def _move_down(self):
        r = self._table.currentRow()
        if r < self._table.rowCount() - 1:
            self._table._swap_rows(r, r + 1)
            self._table.selectRow(r + 1)

    def _update_apply_btn(self):
        self._apply_btn.setEnabled(
            self._yaml_path is not None and
            self._labels_dir is not None
        )

    def _on_apply(self):
        if self._table.rowCount() == 0 and not self._deleted_orig:
            QMessageBox.warning(self, "Nothing to do",
                                "No classes remaining and none deleted.")
            return

        new_names     = self._table.current_names()
        orig_order    = self._table.original_order()

        if any(n == "" for n in new_names):
            QMessageBox.warning(self, "Empty name",
                                "One or more class names are blank.")
            return

        # Build old → new index map.
        # orig_order[new_idx] = old_idx
        # So: old_idx → new_idx
        old_to_new: dict[int, int] = {}
        for new_idx, old_idx in enumerate(orig_order):
            old_to_new[old_idx] = new_idx

        # Build confirmation summary
        changes = []
        for old_idx in sorted(old_to_new):
            new_idx  = old_to_new[old_idx]
            new_name = new_names[new_idx]
            changes.append(f"  class {old_idx} → class {new_idx}  \"{new_name}\"")
        for old_idx in sorted(self._deleted_orig):
            changes.append(f"  class {old_idx} → DELETED")

        msg = (
            f"Labels folder:\n  {self._labels_dir}\n\n"
            f"Changes to apply:\n" + "\n".join(changes) +
            f"\n\nThis will scan every .txt file recursively. "
            f"This cannot be undone. Continue?"
        )
        if QMessageBox.question(
            self, "Confirm", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return

        # Count files for the progress bar
        total = len(list(self._labels_dir.rglob("*.txt")))
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._apply_btn.setEnabled(False)
        self._apply_btn.setText("Working…")
        self._status.setText(f"Scanning {total} label files…")

        self._thread = QThread()
        self._worker = _Worker(
            self._labels_dir, old_to_new, new_names,
            self._yaml_path, set(self._deleted_orig),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_progress(self, done: int, total: int):
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._status.setText(f"Processing file {done} of {total}…")

    def _on_done(self, result: dict):
        self._apply_btn.setEnabled(True)
        self._apply_btn.setText("Apply Changes")
        self._progress.setVisible(False)
        self._deleted_orig.clear()
        self._status.setText("Done.")
        QMessageBox.information(
            self, "Done",
            f"Finished.\n\n"
            f"Files scanned:   {result['files_scanned']}\n"
            f"Files modified:  {result['files_updated']}\n"
            f"Lines remapped:  {result['lines_remapped']}\n"
            f"Lines dropped:   {result['lines_dropped']}  (deleted classes)",
        )

    def _on_error(self, msg: str):
        self._apply_btn.setEnabled(True)
        self._apply_btn.setText("Apply Changes")
        self._progress.setVisible(False)
        self._status.setText("Error.")
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