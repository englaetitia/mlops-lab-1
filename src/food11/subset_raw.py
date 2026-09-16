"""Build a small, git/dvc-friendly copy of the raw Food-11 dataset.

Why this exists
---------------
The full Food-11 download is ~1 GB of JPEGs. Pushing that to a DVC remote is
slow and often times out, so this lab is done with the "reduce the data folder
size" option from the lab sheet:

    data_local/food11_raw/   <- the FULL download, gitignored, never dvc-tracked
    data/food11_raw/         <- N images per category per split, dvc-tracked

The point of the lab is to watch git carry the *pointer* while dvc carries the
*content*; that works exactly the same with 5 images per class as with 5000.

Usage
-----
    uv run python ./src/food11/subset_raw.py                 # 5 per category
    uv run python ./src/food11/subset_raw.py -n 20           # 20 per category
    uv run python ./src/food11/subset_raw.py --random --seed 0
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

# Food-11 file names look like "<category_index>_<image_id>.jpg", e.g. "3_142.jpg".
CATEGORIES: dict[int, str] = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit",
}

SPLITS = ("training", "evaluation", "validation")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data_local" / "food11_raw"
DEFAULT_DEST = PROJECT_ROOT / "data" / "food11_raw"


def category_of(path: Path) -> int | None:
    """Return the category index encoded in the file name, or None if absent."""
    head = path.stem.split("_", 1)[0]
    try:
        index = int(head)
    except ValueError:
        return None
    return index if index in CATEGORIES else None


def group_by_category(split_dir: Path) -> dict[int, list[Path]]:
    """Map category index -> sorted list of image paths inside one split folder."""
    grouped: dict[int, list[Path]] = defaultdict(list)
    # rglob, not glob: some Food-11 mirrors nest the split one level deeper.
    for path in sorted(split_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        index = category_of(path)
        if index is None:
            print(f"  ! skipping unrecognised file name: {path.name}")
            continue
        grouped[index].append(path)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("-n", "--per-category", type=int, default=5,
                        help="images kept per category per split (default: 5)")
    parser.add_argument("--random", action="store_true",
                        help="sample randomly instead of taking the first N by name")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"ERROR: source folder not found: {args.source}")
        print("Put the full Food-11 download at data_local/food11_raw/<split>/ first.")
        return 1

    rng = random.Random(args.seed)

    # Rebuild from scratch so re-running never leaves stale files behind.
    if args.dest.exists():
        shutil.rmtree(args.dest)

    total = 0
    for split in SPLITS:
        split_dir = args.source / split
        if not split_dir.is_dir():
            print(f"! missing split folder, skipping: {split_dir}")
            continue

        grouped = group_by_category(split_dir)
        out_dir = args.dest / split
        out_dir.mkdir(parents=True, exist_ok=True)

        kept_in_split = 0
        for index in sorted(grouped):
            files = grouped[index]
            if args.random:
                chosen = rng.sample(files, min(args.per_category, len(files)))
            else:
                chosen = files[: args.per_category]
            for src in chosen:
                # Flat layout preserved on purpose: the raw dataset keeps the
                # category in the file name, not in the folder structure.
                shutil.copy2(src, out_dir / src.name)
            kept_in_split += len(chosen)

        total += kept_in_split
        print(f"{split:<11} {kept_in_split:>5} images "
              f"({len(grouped)} categories x up to {args.per_category})")

    print(f"\nWrote {total} images to {args.dest}")
    print("Next: dvc add data && git add data.dvc .gitignore && git commit && dvc push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
