"""
core/hef_packager.py  —  TRAINR (orin branch)

Orin-side replacement for the WSL/Hailo ExportWorker.

The AGX Orin cannot run the Hailo Dataflow Compiler (x86-only), so instead of
producing a HEF here we produce a *self-contained bundle* that an x86 machine
can compile without any further hunting for files:

    <model_name>_hef_bundle.zip
    ├── best.onnx
    ├── dataset.yaml        # 'path' rewritten to '.', train/val -> 'calib'
    └── calib/              # N images sampled from the training set
        ├── img001.jpg ... imgNNN.jpg

Because dataset.yaml is rewritten to point at the bundled ./calib folder, the
laptop-side pipeline's get_random_images_from_yaml() resolves the calibration
images *inside the bundle* — no absolute Windows/Orin paths leak across.

Exposes the same log/finished signals as ExportWorker so it is a true drop-in
for the QThread wiring already in tab_onnx.py.
"""
from __future__ import annotations

import random
import shutil
import zipfile
from pathlib import Path

import yaml
from PySide6.QtCore import QObject, Signal

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _sample_calibration_images(yaml_path: str, count: int) -> list[Path]:
    """Sample `count` images from the yaml's training folder.

    Mirrors the resolution logic in export_worker.get_random_images_from_yaml
    (dataset_root = yaml['path'], train_dir = dataset_root / yaml['train'])
    so the Orin samples exactly what the laptop step would have.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data["path"])
    train_dir = dataset_root / data.get("train", "images/train")

    if not train_dir.exists():
        raise RuntimeError(f"Train image directory not found:\n{train_dir}")

    images = [p for p in train_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]

    if len(images) < count:
        raise RuntimeError(
            f"Need at least {count} images for Hailo calibration, "
            f"found {len(images)} in {train_dir}"
        )

    return random.sample(images, count)


class PackagerWorker(QObject):
    """Bundles best.onnx + dataset.yaml + calib images into a portable zip.

    Signal contract is identical to ExportWorker:
        log(str)              — progress lines for the tab's log box
        finished(bool, str)   — (success, message)
    """

    log = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, onnx_path: str, yaml_path: str, output_path: str,
                 model_name: str, sample_count: int = 64):
        super().__init__()
        self.onnx_path = onnx_path
        self.yaml_path = yaml_path
        self.output_path = output_path
        self.model_name = model_name
        self.sample_count = sample_count

    def run(self):
        try:
            onnx_path = Path(self.onnx_path).expanduser().resolve()
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX file not found:\n{onnx_path}")

            model_name = self.model_name.strip() or onnx_path.stem
            bundle_dir = Path(self.output_path) / f"{model_name}_hef_bundle"
            calib_dir = bundle_dir / "calib"

            # Fresh bundle dir each run so stale files never ride along
            if bundle_dir.exists():
                self.log.emit(f"Clearing previous bundle: {bundle_dir}")
                shutil.rmtree(bundle_dir)
            calib_dir.mkdir(parents=True, exist_ok=True)

            self.log.emit("Sampling calibration images…")
            images = _sample_calibration_images(self.yaml_path, self.sample_count)
            for i, img in enumerate(images, 1):
                if i == 1 or i % 16 == 0 or i == len(images):
                    self.log.emit(f"  copied {i}/{len(images)}")
                shutil.copy(img, calib_dir / img.name)

            self.log.emit("Rewriting dataset.yaml for portability…")
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            # Point the yaml at the bundle's own calib folder (relative).
            data["path"] = "."
            data["train"] = "calib"
            data["val"] = "calib"
            with open(bundle_dir / "dataset.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)

            self.log.emit("Copying ONNX…")
            shutil.copy(onnx_path, bundle_dir / "best.onnx")

            self.log.emit("Zipping bundle…")
            zip_path = bundle_dir.with_suffix(".zip")
            if zip_path.exists():
                zip_path.unlink()
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for p in bundle_dir.rglob("*"):
                    z.write(p, p.relative_to(bundle_dir.parent))

            n_classes = len(data.get("names", []) or [])
            self.finished.emit(
                True,
                f"Bundle ready ({len(images)} calib images, "
                f"{n_classes} classes):\n{zip_path}\n\n"
                f"Transfer this zip to the x86 machine and run the HEF compile there."
            )

        except Exception as exc:  # surface any failure to the tab
            self.finished.emit(False, str(exc))
