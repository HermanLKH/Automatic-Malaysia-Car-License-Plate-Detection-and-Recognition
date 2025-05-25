#!/usr/bin/env python3
"""
Generate gt.txt inside the directory pointed to by INPUT_DIR.

Each output line has the form:
    <relative_image_path>\t<label>

Label rules
-----------
1. strip a trailing " (n)" suffix       ->  "BHP7067 (1)"  →  "BHP7067"
2. convert commas to slashes            ->  "T,AA823"      →  "T/AA823"
3. keep '!' tokens exactly as they are  ->  "ABC!1234"     →  "ABC!1234"
"""

import re
import pathlib
import sys

# ──── USER CONFIG ────────────────────────────────────────────────────────────
INPUT_DIR   = 'v4_data_!/test'            # <── change this before you run
GT_FILENAME = 'gt.txt'
VALID_EXT   = ('.jpg', '.jpeg', '.png')
# ─────────────────────────────────────────────────────────────────────────────


def normalise_name(stem: str) -> str:
    """Apply filename ➜ label rules."""
    stem = re.sub(r'\s*\(\d+\)$', '', stem)   # drop " (n)" at end
    stem = stem.replace(',', '/')
    return stem


def generate_gt(folder: pathlib.Path):
    if not folder.exists():
        sys.exit(f"[error] INPUT_DIR not found: {folder}")

    lines = []
    for img_path in sorted(folder.rglob('*')):
        if img_path.suffix.lower() not in VALID_EXT:
            continue
        rel_path = img_path.relative_to(folder)
        label    = normalise_name(img_path.stem)
        lines.append(f"{rel_path.as_posix()}\t{label}")

    out_file = folder / GT_FILENAME
    out_file.write_text('\n'.join(lines), encoding='utf-8')
    print(f"[✓] wrote {out_file}  ({len(lines)} samples)")


if __name__ == '__main__':
    generate_gt(pathlib.Path(INPUT_DIR))
