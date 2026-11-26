import argparse
from itertools import accumulate
from pathlib import Path
import random
import shutil
from typing import TypeVar

T = TypeVar("T")
type Split = tuple[int, int] | tuple[int, int, int]

def face_id(p: Path) -> str:
    return p.name.split("_")[0]  # "00123_Mask_Chin.jpg" -> "00123"

def _split_items(items: list[T], split: Split) -> list[list[T]]:
    total = len(items)
    cuts = [0, *(total * p // 100 for p in accumulate(split))]
    cuts[-1] = total  # guard, in case the percentages don't sum to exactly 100
    return [items[a:b] for a, b in zip(cuts, cuts[1:])] # [0, 700], [700,850], [850,1000]

def split_by_face(
    cmfd_paths: list[Path], imfd_paths: list[Path], split: Split, seed: int, slice: int = 0
) -> tuple[list[list[Path]], list[list[Path]]]:

    # 1. shuffle the unique face IDs (sorted first so the seed is reproducible)
    ids = sorted({face_id(p) for p in cmfd_paths + imfd_paths})
    random.seed(seed)
    random.shuffle(ids)
    if slice:
        ids = ids[:slice]

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
        split_idx = id_to_split.get(face_id(p))
        if split_idx is not None:
            good[split_idx].append(p)
    for p in imfd_paths:
        split_idx = id_to_split.get(face_id(p))
        if split_idx is not None:
            bad[split_idx].append(p)
    return good, bad

def prepare_dataset(dataset_root: Path, split: Split = (70, 15, 15), slice: int = 6000, seed: int = 2026, symlink: bool = False, recreate: bool = False):

    assert len(split) >= 2 and len(split) <= 3, "Split needs to have lenght of 2 or 3"
    assert sum(split) == 100, "Split must sum to 100%"

    dataset_path = (dataset_root.parent / "dataset").resolve()
    # Ensure we start creating the dataset from scratch, so nothing is misplaced
    if dataset_path.exists() and recreate:
        shutil.rmtree(dataset_path)
    elif dataset_path.exists() and not recreate:
        print(f"Dataset at: {dataset_path} already exists, set recreate to true to remove existing dataset and recreate it...")
        return

    dataset_path.mkdir()

    cmfd_paths = sorted([p for p in (dataset_root / "cmfd").rglob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    imfd_paths = sorted([p for p in (dataset_root / "imfd").rglob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")])

    total_size = len(cmfd_paths) + len(imfd_paths)

    y_splits = ["train", "val"] 
    if len(split) == 3:
        y_splits += ["test"]

    # Create the directory structure 
    for y_split in y_splits:
        for y_class in ("good_wear", "bad_wear"):
            (dataset_path / y_split / y_class).resolve().mkdir(parents=True)

    good_wear, bad_wear = split_by_face(cmfd_paths, imfd_paths, split, seed, slice)

    for i, y_split in enumerate(y_splits):
        for y_class, groups in (("good_wear", good_wear), ("bad_wear", bad_wear)):
            for p in groups[i]: # "train" -> 1, all paths
                if symlink:
                    dst = dataset_path / y_split / y_class / p.name
                    dst.symlink_to(p.resolve())
                else:
                    dst = dataset_path / y_split / y_class
                    p.copy_into(dst.resolve())

