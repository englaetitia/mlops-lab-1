"""Check that a processed Food-11 dataset respects the data contract.

Usage
-----
    uv run python src/food11/validate_data.py data/food11_processed_mini
    uv run python src/food11/validate_data.py data/food11_processed --min-per-class 100

Checks (one PASS/FAIL line each):
    1. each split has exactly the 11 class folders
    2. every image opens with PIL, is RGB and is 128x128
    3. every class has at least --min-per-class images in each split
    4. no image appears in two splits (same md5 of the file content)
    5. each class's share of training is within --tau of its share of validation

Exits with status 1 if any check fails, so CI (or a script) can stop on it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

try:  # run from the repo root as a module, or as a script
    from src.food11.data import CLASSES, SPLITS
except ImportError:
    from data import CLASSES, SPLITS

IMAGE_SIZE = (128, 128)


def list_images(root: Path) -> dict[str, dict[str, list[Path]]]:
    """split -> class name -> image files (only folders that exist)."""
    images: dict[str, dict[str, list[Path]]] = {}
    for split in SPLITS:
        split_dir = root / split
        images[split] = {}
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            images[split][class_dir.name] = sorted(
                p for p in class_dir.iterdir() if p.is_file())
    return images


def check_class_folders(images) -> list[str]:
    errors = []
    for split, classes in images.items():
        if not classes:
            errors.append(f"{split}: missing split folder")
            continue
        missing = sorted(set(CLASSES) - set(classes))
        extra = sorted(set(classes) - set(CLASSES))
        if missing:
            errors.append(f"{split}: missing classes {missing}")
        if extra:
            errors.append(f"{split}: unexpected folders {extra}")
    return errors


def check_images(images) -> list[str]:
    errors = []
    for classes in images.values():
        for files in classes.values():
            for path in files:
                try:
                    with Image.open(path) as img:
                        img.load()  # really decode it, not just read the header
                        if img.mode != "RGB":
                            errors.append(f"{path}: mode {img.mode}, expected RGB")
                        if img.size != IMAGE_SIZE:
                            errors.append(f"{path}: size {img.size}, expected {IMAGE_SIZE}")
                except Exception as exc:  # any file PIL cannot read
                    errors.append(f"{path}: cannot open ({exc})")
    return errors


def check_min_per_class(images, min_per_class: int) -> list[str]:
    errors = []
    for split, classes in images.items():
        for name in CLASSES:
            count = len(classes.get(name, []))
            if count < min_per_class:
                errors.append(f"{split}/{name}: {count} images (< {min_per_class})")
    return errors


def check_no_leakage(images) -> list[str]:
    """The same file content must not be in two different splits."""
    seen: dict[str, set[str]] = defaultdict(set)
    where: dict[str, list[Path]] = defaultdict(list)
    for split, classes in images.items():
        for files in classes.values():
            for path in files:
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                seen[digest].add(split)
                where[digest].append(path)
    errors = []
    for digest, splits in seen.items():
        if len(splits) > 1:
            paths = ", ".join(str(p) for p in where[digest])
            errors.append(f"duplicate in {sorted(splits)}: {paths}")
    return errors


def check_balance(images, tau: float) -> list[str]:
    """Class shares in training and validation should be close."""
    train = {name: len(images["training"].get(name, [])) for name in CLASSES}
    val = {name: len(images["validation"].get(name, [])) for name in CLASSES}
    n_train, n_val = sum(train.values()), sum(val.values())
    if n_train == 0 or n_val == 0:
        return ["training or validation split is empty"]
    errors = []
    for name in CLASSES:
        share_train = train[name] / n_train
        share_val = val[name] / n_val
        if abs(share_train - share_val) > tau:
            errors.append(f"{name}: {share_train:.1%} of training vs "
                          f"{share_val:.1%} of validation (tau={tau})")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path,
                        help="e.g. data/food11_processed_mini")
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()

    if not args.dataset.is_dir():
        print(f"FAIL  dataset folder not found: {args.dataset}")
        return 1

    images = list_images(args.dataset)
    checks = [
        ("1. 11 class folders in each split", check_class_folders(images)),
        ("2. images open, RGB, 128x128", check_images(images)),
        (f"3. at least {args.min_per_class} images per class",
         check_min_per_class(images, args.min_per_class)),
        ("4. no image in two splits", check_no_leakage(images)),
        (f"5. class balance train vs val (tau={args.tau})",
         check_balance(images, args.tau)),
    ]

    total = sum(len(f) for c in images.values() for f in c.values())
    print(f"dataset: {args.dataset}  ({total} images)\n")
    failed = False
    for title, errors in checks:
        print(f"{'PASS' if not errors else 'FAIL'}  {title}")
        for error in errors[:10]:  # don't flood the terminal
            print(f"        - {error}")
        if len(errors) > 10:
            print(f"        ... and {len(errors) - 10} more")
        failed = failed or bool(errors)

    print("\nRESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
