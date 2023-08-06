import torch
import cv2
from torchvision import datasets
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout import CoarseDropout, cutout
from PIL import Image

NORM_DATA_MEAN = (0.49139968, 0.48215841, 0.44653091)
NORM_DATA_STD = (0.24703223, 0.24348513, 0.26158784)
CIFAR_CLASS_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
    ]
TRAIN_TRANSFORM = A.Compose([
    A.Normalize(
        mean=NORM_DATA_MEAN,
        std=NORM_DATA_STD,
    ),
    A.HorizontalFlip(),
    A.Compose([
        A.PadIfNeeded(min_height=40, min_width=40, p=1.0),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16,
            min_holes=1, min_height=16, min_width=16, 
            fill_value=NORM_DATA_MEAN, mask_fill_value=None, p=1.0),
        A.RandomCrop(p=1.0, height=32, width=32)
    ]),
    ToTensorV2(),
])
TEST_TRANSFORM = A.Compose([
    A.Normalize(
        mean=NORM_DATA_MEAN,
        std=NORM_DATA_STD,
    ),
    ToTensorV2(),
])

class CifarAlbumentationsDataset(datasets.CIFAR10):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform:
            augmented = self.transform(image=img)
            image = augmented['image']
        return image, target

