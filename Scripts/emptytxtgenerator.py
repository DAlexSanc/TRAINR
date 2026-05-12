# Python version
import os
from pathlib import Path

img_dir = Path("C:/Users/Diego/Documents/SWT_RAWDATA/Benchmark/not_missing")
lbl_dir = Path("C:/Users/Diego\Documents/SWT_RAWDATA/Benchmark/not_missing")

for img in img_dir.glob("*.png"):
    label = lbl_dir / img.with_suffix(".txt").name
    if not label.exists():
        label.touch()