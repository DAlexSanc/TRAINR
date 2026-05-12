"""
dataset_analyzer.py
--------------------
Pure-Python backend for YOLO/LabelMe dataset analysis.
Supports .txt (YOLO) and .json (LabelMe) label formats.
Recursively scans subfolders.
No GUI dependency — safe to import in tests or CLI scripts.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetStats:
    label_dir: Path
    class_names: list[str]          # ordered list; index == YOLO class id

    total_images: int = 0
    images_with_labels: int = 0
    empty_images: int = 0

    # Keys are YOLO integer class ids (for .txt) or auto-assigned ints (for .json)
    class_counts: Counter = field(default_factory=Counter)
    instances_per_image: list[int] = field(default_factory=list)

    # ── derived ───────────────────────────────────────────────────────────────

    @property
    def total_instances(self) -> int:
        return sum(self.class_counts.values())

    @property
    def avg_instances_per_labeled_image(self) -> float:
        if not self.instances_per_image:
            return 0.0
        return sum(self.instances_per_image) / len(self.instances_per_image)

    @property
    def labeled_pct(self) -> float:
        return 100 * self.images_with_labels / self.total_images if self.total_images else 0.0

    @property
    def empty_pct(self) -> float:
        return 100 * self.empty_images / self.total_images if self.total_images else 0.0

    def class_name(self, idx: int) -> str:
        if 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return f"class {idx}"

    def class_pct(self, idx: int) -> float:
        total = self.total_instances
        return 100 * self.class_counts.get(idx, 0) / total if total else 0.0

    def sorted_class_ids(self) -> list[int]:
        return sorted(self.class_counts.keys())


# ──────────────────────────────────────────────────────────────────────────────
# dataset.yaml auto-discovery
# ──────────────────────────────────────────────────────────────────────────────

def _find_yaml_names(start: Path) -> list[str] | None:
    """
    Walk upward from `start` looking for a dataset.yaml / dataset.yml that
    contains a 'names:' block.  Returns an ordered list of class names if
    found, else None.
    """
    for folder in [start, *start.parents]:
        for stem in ("dataset.yaml", "dataset.yml", "data.yaml", "data.yml"):
            candidate = folder / stem
            if not candidate.exists():
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore")
                names = _parse_yaml_names(text)
                if names:
                    return names
            except OSError:
                pass
        # Stop after a reasonable depth (don't crawl to filesystem root)
        if folder == start.anchor or len(folder.parts) <= 2:
            break
    return None


def _parse_yaml_names(text: str) -> list[str]:
    """
    Minimal YAML parser for the 'names:' block — no pyyaml dependency.

    Handles both forms:
        names: [cat, dog, bird]
        names:
          0: cat
          1: dog
    """
    lines = text.splitlines()
    in_names = False
    names: dict[int, str] = {}

    for line in lines:
        stripped = line.strip()

        # Single-line list form:  names: [cat, dog]
        if stripped.lower().startswith("names:") and "[" in stripped:
            inner = stripped.split("[", 1)[1].split("]")[0]
            return [n.strip().strip("'\"") for n in inner.split(",") if n.strip()]

        # Block form start
        if stripped.lower().startswith("names:"):
            in_names = True
            continue

        if in_names:
            if stripped == "" or (not line.startswith(" ") and not line.startswith("\t")):
                in_names = False
                continue
            # expect "  0: cat" or "  - cat"
            if ":" in stripped:
                idx_str, _, name = stripped.partition(":")
                try:
                    names[int(idx_str.strip())] = name.strip().strip("'\"")
                except ValueError:
                    pass
            elif stripped.startswith("-"):
                idx = len(names)
                names[idx] = stripped.lstrip("- ").strip().strip("'\"")

    if names:
        return [names[i] for i in sorted(names)]
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Per-file parsers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_txt(path: Path) -> list[int]:
    """Return list of integer class ids from a YOLO .txt label file."""
    ids = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if parts:
            try:
                ids.append(int(parts[0]))
            except ValueError:
                pass
    return ids


def _parse_json(path: Path, name_to_id: dict[str, int]) -> tuple[list[int], set[str]]:
    """
    Parse a LabelMe JSON file.

    Returns
    -------
    ids:       list of integer class ids for instances in this file
    new_names: set of label strings not yet in name_to_id (caller adds them)
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, OSError):
        return [], set()

    ids: list[int] = []
    new_names: set[str] = set()

    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip()
        if not label:
            continue
        if label not in name_to_id:
            new_names.add(label)
        else:
            ids.append(name_to_id[label])

    return ids, new_names


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def analyze_dataset(
    label_dir: Path | str,
    class_names: list[str] | None = None,
) -> DatasetStats:
    """
    Recursively scan a directory tree for YOLO .txt and/or LabelMe .json
    label files and return a populated DatasetStats object.

    Class name resolution order (highest priority first):
      1. `class_names` argument passed in by the caller
      2. dataset.yaml / data.yaml found anywhere in the directory tree
      3. Labels extracted from LabelMe JSON shapes (sorted alphabetically)
      4. Fallback: 'class N' for any unmapped YOLO integer index
    """
    label_dir = Path(label_dir)

    # ── collect all label files recursively ──────────────────────────────────
    txt_files  = sorted(label_dir.rglob("*.txt"))
    json_files = sorted(label_dir.rglob("*.json"))

    # ── resolve class names ───────────────────────────────────────────────────
    if class_names:
        resolved_names = list(class_names)
    else:
        resolved_names = _find_yaml_names(label_dir) or []

    # Build mutable name→id map; will grow if JSON files introduce new labels
    name_to_id: dict[str, int] = {n: i for i, n in enumerate(resolved_names)}

    # ── two-pass JSON scan if we have no names yet ────────────────────────────
    # First pass: collect every unique label name so the id assignment is stable
    if not name_to_id and json_files:
        all_json_labels: set[str] = set()
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
                for shape in data.get("shapes", []):
                    lbl = shape.get("label", "").strip()
                    if lbl:
                        all_json_labels.add(lbl)
            except (json.JSONDecodeError, OSError):
                pass
        resolved_names = sorted(all_json_labels)
        name_to_id = {n: i for i, n in enumerate(resolved_names)}

    # ── build stats object ────────────────────────────────────────────────────
    stats = DatasetStats(label_dir=label_dir, class_names=list(resolved_names))

    # ── process .txt files ────────────────────────────────────────────────────
    for txt_file in txt_files:
        stats.total_images += 1
        ids = _parse_txt(txt_file)
        if not ids:
            stats.empty_images += 1
        else:
            stats.images_with_labels += 1
            stats.instances_per_image.append(len(ids))
            for cls in ids:
                stats.class_counts[cls] += 1

    # ── process .json files ───────────────────────────────────────────────────
    for json_file in json_files:
        stats.total_images += 1

        ids, new_names = _parse_json(json_file, name_to_id)

        # Register any brand-new label names encountered
        for name in sorted(new_names):
            new_id = len(name_to_id)
            name_to_id[name] = new_id
            stats.class_names.append(name)

        # Re-parse now that name_to_id is complete for new names
        if new_names:
            ids, _ = _parse_json(json_file, name_to_id)

        if not ids:
            stats.empty_images += 1
        else:
            stats.images_with_labels += 1
            stats.instances_per_image.append(len(ids))
            for cls in ids:
                stats.class_counts[cls] += 1

    return stats