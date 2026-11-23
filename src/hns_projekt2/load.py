from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import datasets, transforms
import numpy as np

from projekt2.config import IMAGENET_MEAN, IMAGENET_STD

class MaskedFaceDataset(Dataset):
    def __init__(self, paths, labels):
        assert len(paths) == len(labels), "paths and labels must have same length"
        self.paths, self.labels = paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            image = im.convert("RGB")
        return image, torch.tensor(self.labels[idx], dtype=torch.long)

def build_path_dataset(root: Path, transform=None) -> Dataset: # maybe add image transforms later
    paths, labels = [], []
    for p in sorted((root / "CMFD").glob("*.jpg")):
        paths.append(p)
        labels.append(0)
    for p in sorted((root / "IMFD").rglob("*.jpg")):
        paths.append(p)
        labels.append(1)
    return MaskedFaceDataset(paths, labels)

def make_transforms(augment: bool = True, image_size: int = 224):
    train_ops = []
    if augment:
        train_ops += [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.12),
        ]
    else:
        train_ops += [transforms.Resize((image_size, image_size))]
    train_ops += [transforms.ToTensor(),
                  transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    eval_ops = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(train_ops), eval_ops

def build_loader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0, pin_memory: bool | None = None) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory or torch.cuda.is_available()
    )

def loaders(paths, labels, splits, batch_size: int = 32, num_workers: int = 0):
    train_idx, val_idx, test_idx = splits
    train_ds = MaskedFaceDataset([paths[i] for i in train_idx], labels[train_idx])
    val_ds = MaskedFaceDataset([paths[i] for i in val_idx], labels[val_idx])
    test_ds = MaskedFaceDataset([paths[i] for i in test_idx],labels[test_idx]) 
    return (
        DataLoader(train_ds, batch_size=batch_size,
                   shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size,
                   shuffle=False, num_workers=num_workers),
    )
