"""
ui/sidebar.py  —  TRAINR
Sidebar: config fields + model radio list + dataset tool links
"""
from __future__ import annotations

from PySide6.QtCore    import Qt, Signal
from PySide6.QtGui     import QColor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ui.widgets import hsep, link_btn, section_label


class Sidebar(QWidget):
    model_changed = Signal(int)

    _MODEL_ITEMS = [
        ("Detection Nano",     "nano"),
        ("Detection Small",    "small"),
        ("Detection Medium",   "medium"),
        ("Detection Large",    "large"),
        ("Detection XLarge",   "xlarge"),
        ("Segmentation Nano",  "nano"),
        ("Segmentation Small", "small"),
        ("Segmentation Medium","medium"),
        ("Segmentation Large", "large"),
        ("Segmentation XLarge","xlarge"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Scrollable content ────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner     = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(10, 12, 10, 8)
        inner_lay.setSpacing(0)

        # Config
        inner_lay.addWidget(section_label("Configuration"))
        inner_lay.addSpacing(5)

        lbl_yaml = QLabel("Dataset YAML")
        lbl_yaml.setStyleSheet("font-size: 9pt; color: #706B63;")
        inner_lay.addWidget(lbl_yaml)
        inner_lay.addSpacing(2)

        row1 = QHBoxLayout()
        row1.setSpacing(3)
        self.dataset_linedit = QLineEdit()
        self.dataset_linedit.setPlaceholderText("dataset.yaml")
        row1.addWidget(self.dataset_linedit)
        self.dataset_button = QPushButton("…")
        self.dataset_button.setFixedSize(26, 26)
        self.dataset_button.setObjectName("iconBtn")
        row1.addWidget(self.dataset_button)
        inner_lay.addLayout(row1)
        inner_lay.addSpacing(8)

        lbl_out = QLabel("Output folder")
        lbl_out.setStyleSheet("font-size: 9pt; color: #706B63;")
        inner_lay.addWidget(lbl_out)
        inner_lay.addSpacing(2)

        row2 = QHBoxLayout()
        row2.setSpacing(3)
        self.output_linedit = QLineEdit()
        self.output_linedit.setPlaceholderText("output directory")
        row2.addWidget(self.output_linedit)
        self.output_button = QPushButton("…")
        self.output_button.setFixedSize(26, 26)
        self.output_button.setObjectName("iconBtn")
        row2.addWidget(self.output_button)
        inner_lay.addLayout(row2)
        inner_lay.addSpacing(12)

        inner_lay.addWidget(hsep())
        inner_lay.addSpacing(10)

        # Model list
        inner_lay.addWidget(section_label("Models"))
        inner_lay.addSpacing(4)

        self._model_list = QListWidget()
        self._model_list.setSpacing(0)
        self._model_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        for name, tag in self._MODEL_ITEMS:
            item = QListWidgetItem(f"  {name}")
            item.setData(Qt.ItemDataRole.UserRole, tag)
            self._model_list.addItem(item)

        sep_item = QListWidgetItem("─" * 22)
        sep_item.setFlags(Qt.ItemFlag.NoItemFlags)
        sep_item.setForeground(QColor("#555"))
        self._model_list.insertItem(5, sep_item)
        self._model_list.insertItem(6, QListWidgetItem())
        self._model_list.item(6).setFlags(Qt.ItemFlag.NoItemFlags)

        self._model_list.setCurrentRow(1)
        self._model_list.itemClicked.connect(self._on_model_click)
        self._model_list.setMaximumHeight(280)
        inner_lay.addWidget(self._model_list)
        inner_lay.addStretch()

        scroll.setWidget(inner)
        root.addWidget(scroll, stretch=1)

        # ── Dataset tools footer ───────────────────────────────────────────────
        footer     = QWidget()
        footer.setFixedHeight(150)
        footer_lay = QVBoxLayout(footer)
        footer_lay.setContentsMargins(12, 6, 12, 8)
        footer_lay.setSpacing(4)

        footer_lay.addWidget(hsep())
        footer_lay.addSpacing(0)
        footer_lay.addWidget(section_label("Dataset tools"))

        links_col = QVBoxLayout()
        links_col.setSpacing(2)
        links_col.setContentsMargins(0, 0, 0, 0)

        self.analyze_btn      = link_btn("Analyze")
        self.class_rename_btn = link_btn("Rename Classes")
        self.organize_btn     = link_btn("Organize")
        self.emptylabels_btn  = link_btn("Empty labels")

        for i, btn in enumerate([self.analyze_btn, self.class_rename_btn,
                                  self.organize_btn, self.emptylabels_btn]):
            links_col.addWidget(btn)
            if i < 3:
                sep = hsep()
                sep.setStyleSheet("background: #C4BFB5;")
                sep.setFixedHeight(2)
                sep.setContentsMargins(5, 0, 5, 0)
                links_col.addWidget(sep)

        footer_lay.addLayout(links_col)
        root.addWidget(footer)

    def _on_model_click(self, item: QListWidgetItem):
        if not (item.flags() & Qt.ItemFlag.ItemIsSelectable):
            return
        row  = self._model_list.row(item)
        flat = row if row < 5 else row - 2
        self.model_changed.emit(flat)

    def current_model_index(self) -> int:
        row = self._model_list.currentRow()
        return row if row < 5 else max(0, row - 2)

    def set_model_index(self, idx: int):
        self._model_list.setCurrentRow(idx if idx < 5 else idx + 2)
