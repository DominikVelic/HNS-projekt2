import argparse
from pathlib import Path

from hns_projekt2.train.data.yolo import prepare_dataset_for_yolo

type Split = tuple[int, int] | tuple[int, int, int]

def match_split(args: str) -> Split:
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
    parser.add_argument('--recreate', action="store_true", help="Set if you want to remove an old yolo dataset and recreate it from scrach")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    d_split = match_split(args.split)
    prepare_dataset_for_yolo(dataset_root, d_split, args.seed, args.symlink, args.recreate)

if __name__ == "__main__":
    main()


