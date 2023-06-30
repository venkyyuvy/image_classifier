import torch
import cv2
from torchvision import datasets
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout import CoarseDropout
from PIL import Image

RAW_DATA_MEAN = (125.30691805, 122.95039414, 113.86538318)
NORM_DATA_MEAN = (0.49139968, 0.48215841, 0.44653091)
NORM_DATA_STD = (0.24703223, 0.24348513, 0.26158784)

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
        A.HorizontalFlip(p=0.5,),
        A.ShiftScaleRotate(
            shift_limit=0.06, scale_limit=0.3, rotate_limit=45, p=0.3),
        CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1,
            min_height=16, min_width=16, p=0.3,
            fill_value=RAW_DATA_MEAN,
            mask_fill_value = None),
        A.Normalize(
            mean=NORM_DATA_MEAN,
            std=NORM_DATA_STD,
        ),
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
