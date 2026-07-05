"""
core/dataset_analyzer.py  —  TRAINR
Pure-Python backend for YOLO/LabelMe dataset analysis.
No GUI dependency.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetStats:
    label_dir:   Path
    class_names: list[str]

    total_images:      int = 0
    images_with_labels:int = 0
    empty_images:      int = 0
    class_counts: Counter  = field(default_factory=Counter)
    instances_per_image: list[int] = field(default_factory=list)

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
        return 100 * self.images_with_labels / self.total_images \
               if self.total_images else 0.0

    @property
    def empty_pct(self) -> float:
        return 100 * self.empty_images / self.total_images \
               if self.total_images else 0.0

    def class_name(self, idx: int) -> str:
        if 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return f"class {idx}"

    def class_pct(self, idx: int) -> float:
        total = self.total_instances
        return 100 * self.class_counts.get(idx, 0) / total if total else 0.0

    def sorted_class_ids(self) -> list[int]:
        return sorted(self.class_counts.keys())


def _find_yaml_names(start: Path) -> list[str] | None:
    for folder in [start, *start.parents]:
        for stem in ("dataset.yaml", "dataset.yml", "data.yaml", "data.yml"):
            candidate = folder / stem
            if not candidate.exists():
                continue
            try:
                text  = candidate.read_text(encoding="utf-8", errors="ignore")
                names = _parse_yaml_names(text)
                if names:
                    return names
            except OSError:
                pass
        if folder == start.anchor or len(folder.parts) <= 2:
            break
    return None


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


def _parse_txt(path: Path) -> list[int]:
    ids = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if parts:
            try:
                ids.append(int(parts[0]))
            except ValueError:
                pass
    return ids


def _parse_json(path: Path,
                name_to_id: dict[str, int]) -> tuple[list[int], set[str]]:
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


def analyze_dataset(label_dir: Path | str,
                    class_names: list[str] | None = None) -> DatasetStats:
    label_dir  = Path(label_dir)
    txt_files  = sorted(label_dir.rglob("*.txt"))
    json_files = sorted(label_dir.rglob("*.json"))

    resolved_names = list(class_names) if class_names \
                     else (_find_yaml_names(label_dir) or [])
    name_to_id: dict[str, int] = {n: i for i, n in enumerate(resolved_names)}

    if not name_to_id and json_files:
        all_labels: set[str] = set()
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
                for shape in data.get("shapes", []):
                    lbl = shape.get("label", "").strip()
                    if lbl:
                        all_labels.add(lbl)
            except (json.JSONDecodeError, OSError):
                pass
        resolved_names = sorted(all_labels)
        name_to_id     = {n: i for i, n in enumerate(resolved_names)}

    stats = DatasetStats(label_dir=label_dir, class_names=list(resolved_names))

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

    for json_file in json_files:
        stats.total_images += 1
        ids, new_names = _parse_json(json_file, name_to_id)

        for name in sorted(new_names):
            new_id = len(name_to_id)
            name_to_id[name] = new_id
            stats.class_names.append(name)

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
