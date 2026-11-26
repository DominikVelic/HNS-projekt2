from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def make_loaders(data_root: Path, cfg: dict, batch_size: int, num_workers: int):
    # Each pretrained model has its own input size and normalization
    # (YOLO uses no mean/std, MobileViT uses 256px), so cfg comes from build_model.
    train_tf = timm.data.create_transform(
        **cfg, is_training=True,
        scale=(0.75, 1.0),   # timm's default (0.08, 1) can crop the mask out of the face
        hflip=0.5,
        color_jitter=0.15,
    )
    eval_tf = timm.data.create_transform(**cfg, is_training=False)

    loaders, classes = {}, None
    for split in ("train", "val", "test"):
        if not (data_root / split).exists():
            continue
        ds = ImageFolder(data_root / split, transform=train_tf if split == "train" else eval_tf)
        classes = ds.classes  # ['bad_wear', 'good_wear'], same order YOLO uses
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
    return loaders, classes
