from torchvision import transforms

from hns_projekt2.train.config import IMAGENET_MEAN, IMAGENET_STD


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

    eval_ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)
