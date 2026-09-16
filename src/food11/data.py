"""Prepare the Food-11 images for a torchvision/ResNet training pipeline.

What ResNet (via torchvision.datasets.ImageFolder) expects
---------------------------------------------------------
ImageFolder infers the label from the *directory name*, so the tree has to be

    <root>/<split>/<class name>/<image files>

The raw Food-11 download is flat and encodes the class in the file name
("3_142.jpg" -> category 3 -> "Egg"), which ImageFolder cannot read. This
script fixes that, and at the same time shrinks every image to 128x128 so the
dataset is small enough to iterate on quickly.

Inputs / outputs
----------------
    data/food11_raw/<split>/<category>_<id>.jpg            (input)
      -> data/food11_processed/<split>/<Category>/<file>   (all images, 128x128)
      -> data/food11_processed_mini/<split>/<Category>/...  (<= 100 per category)

The *_mini dataset exists so you can run the training code end to end in
seconds and be sure the pipeline is correct before spending time on the full
dataset.

Usage
-----
    uv run python ./src/food11/data.py
    uv run python ./src/food11/data.py --source data_local/food11_raw
    uv run python ./src/food11/data.py --size 224 --mini-per-category 50
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

# Category index -> human readable name. The index is the first part of the
# raw file name, and the name becomes the ImageFolder class directory.
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
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SOURCE = DATA_DIR / "food11_raw"
DEFAULT_PROCESSED = DATA_DIR / "food11_processed"
DEFAULT_MINI = DATA_DIR / "food11_processed_mini"

DEFAULT_SIZE = 128
DEFAULT_MINI_PER_CATEGORY = 100


def category_of(path: Path) -> int | None:
    """Read the category index out of a raw Food-11 file name."""
    head = path.stem.split("_", 1)[0]
    try:
        index = int(head)
    except ValueError:
        return None
    return index if index in CATEGORIES else None


def group_by_category(split_dir: Path) -> dict[int, list[Path]]:
    """Map category index -> sorted image paths for one split folder."""
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(split_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        index = category_of(path)
        if index is None:
            print(f"  ! unrecognised file name, skipped: {path.name}")
            continue
        grouped[index].append(path)
    return grouped


def resize_to(src: Path, dst: Path, size: int) -> None:
    """Write src to dst as a `size` x `size` RGB JPEG."""
    with Image.open(src) as image:
        # JPEG cannot store an alpha channel, and ResNet wants 3 channels.
        image = image.convert("RGB")
        # LANCZOS is the best-quality downscaling filter Pillow offers.
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst, format="JPEG", quality=90)


def build(source: Path, processed: Path, mini: Path, size: int,
          mini_per_category: int) -> int:
    # Remove previous runs so the output is a pure function of the input; a
    # stale image left behind would silently change the dvc hash.
    for out_root in (processed, mini):
        if out_root.exists():
            shutil.rmtree(out_root)

    total = 0
    for split in SPLITS:
        split_dir = source / split
        if not split_dir.is_dir():
            print(f"! missing split folder, skipped: {split_dir}")
            continue

        grouped = group_by_category(split_dir)
        split_count = 0

        for index in sorted(grouped):
            name = CATEGORIES[index]
            files = grouped[index]

            for rank, src in enumerate(files):
                # ".jpg" regardless of the input extension: every output image
                # is re-encoded as JPEG above.
                target = processed / split / name / f"{src.stem}.jpg"
                resize_to(src, target, size)

                if rank < mini_per_category:
                    mini_target = mini / split / name / target.name
                    mini_target.parent.mkdir(parents=True, exist_ok=True)
                    # Already the right size and format - copying beats
                    # decoding and re-encoding a second time.
                    shutil.copy2(target, mini_target)

            split_count += len(files)
            print(f"  {split:<11} {name:<16} {len(files):>5} images "
                  f"({min(len(files), mini_per_category)} in mini)")

        total += split_count
        print(f"{split}: {split_count} images across {len(grouped)} categories\n")

    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="raw dataset root (default: data/food11_raw)")
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--mini", type=Path, default=DEFAULT_MINI)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help="output edge length in pixels (default: 128)")
    parser.add_argument("--mini-per-category", type=int,
                        default=DEFAULT_MINI_PER_CATEGORY,
                        help="max images per category in the mini dataset")
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"ERROR: raw dataset not found at {args.source}")
        print("Run `dvc pull` first, or pass --source data_local/food11_raw.")
        return 1

    print(f"source : {args.source}")
    print(f"outputs: {args.processed}\n         {args.mini}")
    print(f"size   : {args.size}x{args.size}\n")

    total = build(args.source, args.processed, args.mini,
                  args.size, args.mini_per_category)

    if total == 0:
        print("ERROR: no images were processed - check the source folder.")
        return 1

    print(f"Done: {total} images processed.")
    print("Next: dvc add data && git add data.dvc && git commit && dvc push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
