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


class CifarAlbumentationsDataset(datasets.CIFAR10):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform:
            augmented = self.transform(image=img)
            image = augmented['image']
        return image, target

def prepare_cifar_dataloader(
    data_path='../../data', batch_size=512, seed=1):
    train_transform = A.Compose([
        A.Normalize(
            mean=NORM_DATA_MEAN,
            std=NORM_DATA_STD,
        ),
        A.HorizontalFlip(),
        A.Compose([
            A.PadIfNeeded(min_height=40, min_width=40, p=1.0),
            A.RandomCrop(p=1.0, height=32, width=32)
        ]),
        cutout.Cutout(num_holes=1, max_h_size=8, max_w_size=8,
                      fill_value=NORM_DATA_MEAN, p=0.5),
        ToTensorV2(),
    ])
    test_transform = A.Compose([
        A.Normalize(
            mean=NORM_DATA_MEAN,
            std=NORM_DATA_STD,
        ),
        ToTensorV2(),
    ])
    train_data = CifarAlbumentationsDataset(
            root=data_path,
            train=True, download=True,
            transform=train_transform)
    test_data = CifarAlbumentationsDataset(
            data_path, train=False, download=True, transform=test_transform)

    cuda = torch.cuda.is_available()

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    dataloader_args = dict(
        shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)
    return train_loader, test_loader
