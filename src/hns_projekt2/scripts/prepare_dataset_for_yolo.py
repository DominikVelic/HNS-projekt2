
import argparse
from itertools import accumulate
from pathlib import Path
import random
import shutil
from typing import TypeVar

type Split = tuple[int, int] | tuple[int, int, int]

T = TypeVar("T")

def face_id(p: Path) -> str:
    return p.name.split("_")[0]  # "00123_Mask_Chin.jpg" -> "00123"

def _split_items(items: list[T], split: Split) -> list[list[T]]:
    total = len(items)
    cuts = [0, *(total * p // 100 for p in accumulate(split))]
    cuts[-1] = total  # guard, in case the percentages don't sum to exactly 100
    return [items[a:b] for a, b in zip(cuts, cuts[1:])] # [0, 700], [700,850], [850,1000]

def split_by_face(
    cmfd_paths: list[Path], imfd_paths: list[Path], split: Split, seed: int
) -> tuple[list[list[Path]], list[list[Path]]]:

    # 1. shuffle the unique face IDs (sorted first so the seed is reproducible)
    ids = sorted({face_id(p) for p in cmfd_paths + imfd_paths})
    random.seed(seed)
    random.shuffle(ids)

    # 2. split the IDs, then map each ID to its split index, 0=train, 1=val, 2=test
    id_to_split = {
        fid: i
        for i, part in enumerate(_split_items(ids, split)) 
        for fid in part
    }

    # 3. send every path to the split its face ID belongs to
    n = len(split)
    good: list[list[Path]] = [[] for _ in range(n)] # [[train],[val],[test]]
    bad: list[list[Path]] = [[] for _ in range(n)]
    for p in cmfd_paths:
        split_idx = id_to_split[face_id(p)] 
        good[split_idx].append(p)
    for p in imfd_paths:
        split_idx = id_to_split[face_id(p)] 
        bad[split_idx].append(p)
    return good, bad

def prepare_dataset_for_yolo(dataset_root: Path, split: Split = (70, 15, 15), seed: int = 2026, symlink: bool = False):

    assert len(split) >= 2 and len(split) <= 3, "Split needs to have lenght of 2 or 3"
    assert sum(split) == 100, "Split must sum to 100%"

    cmfd_paths = sorted([p for p in (dataset_root / "cmfd").rglob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    imfd_paths = sorted([p for p in (dataset_root / "imfd").rglob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    total_size = len(cmfd_paths) + len(imfd_paths)

    yolo_dataset_path = (dataset_root.parent / "yolod").resolve()

    # Ensure we start creating the yolo dataset from scratch, so nothing is misplaced
    if yolo_dataset_path.exists():
        shutil.rmtree(yolo_dataset_path)

    yolo_dataset_path.mkdir()

    y_splits = ["train", "val"] 
    if len(split) == 3:
        y_splits += ["test"]

    # Create the directory structure 
    for y_split in y_splits:
        for y_class in ("good_wear", "bad_wear"):
            (yolo_dataset_path / y_split / y_class).resolve().mkdir(parents=True)

    good_wear, bad_wear = split_by_face(cmfd_paths, imfd_paths, split, seed)

    for i, y_split in enumerate(y_splits):
        for y_class, groups in (("good_wear", good_wear), ("bad_wear", bad_wear)):
            for p in groups[i]: # "train" -> 1, all paths
                if symlink:
                    dst = yolo_dataset_path / y_split / y_class / p.name
                    dst.symlink_to(p.resolve())
                else:
                    dst = yolo_dataset_path / y_split / y_class
                    p.copy_into(dst.resolve())

def _match_split(args: str) -> Split:
    arg_l = args.split(",")
    nums = [int(a) for a in arg_l]
    match nums:
        case [train, val]:
            return (train, val)
        case [train, val, test]:
            return (train, val, test)
        case _:
            raise SystemExit(f"split needs 2 or 3 ints, got {len(nums)}: {args}")

def main():

    parser = argparse.ArgumentParser(
                    prog='Yolo dataset preparation',
                    description='Prepare dataset for yolo usage, splits image paths into good_wear and bad_wear classes with a split passed in arguments',
                    )

    parser.add_argument('-d', '--dataset_root', type=str, required=True)
    parser.add_argument('-s', '--split', help="Should be in form of 70,15,15 or 70,15 for example", type=str, default="70,15,15")
    parser.add_argument('--seed', default=2026, type=int)
    parser.add_argument('--symlink', action="store_true", help="Only symlink images, instead of copying to save disk space")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    d_split = _match_split(args.split)
    prepare_dataset_for_yolo(dataset_root, d_split, args.seed, args.symlink)

if __name__ == "__main__":
    main()


