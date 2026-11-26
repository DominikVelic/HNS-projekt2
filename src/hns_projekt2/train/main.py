import argparse
from pathlib import Path
from hns_projekt2.train.train import MODELS_KEYS, train_model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune models on the yolod dataset")
    parser.add_argument("-d", "--data", type=str, required=True)
    parser.add_argument("-r", "--repeats", type=int, default=5)
    parser.add_argument("-m", "--model", choices=[*MODELS_KEYS, "all"], default="all")
    parser.add_argument("-o", "--out", type=str, default="runs/")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    keys = list(MODELS_KEYS) if args.model == "all" else [args.model]
    for key in keys:
        for i in range(args.repeats):
            seed = args.seed + i
            print(f"\n===Repeat: {i} {key} (seed {seed})===")
            train_model(key, Path(args.data).resolve(), Path(args.out) / key / f"run_{i}",
                        args.epochs, args.batch_size, args.lr, args.workers, seed)


if __name__ == "__main__":
    main()
