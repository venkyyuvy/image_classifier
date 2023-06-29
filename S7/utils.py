import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets
from torchvision.datasets import MNIST

class MNISTAlbumentationsDataset(MNIST):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
    def __getitem__(self, idx):
        image, label = self.data[idx], int(self.targets[idx])
        image = Image.fromarray(image.numpy(), mode="L")
        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = augmented['image']
        return image, label


def prepare_mnist_data(train_transforms, test_transforms,
    data_path='../data', batch_size=512, seed=1, is_alb_transforms=False):
    MNISTdataset = MNISTAlbumentationsDataset if is_alb_transforms else datasets.MNIST
    print(MNISTdataset)
    train_data = MNISTdataset(
            root=data_path,
            train=True, download=True, transform=train_transforms)
    test_data = MNISTdataset(
            data_path, train=False, download=True, transform=test_transforms)

    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(
        shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)

    return train_loader, test_loader

def plot_img_batch(train_loader, n_img=12):
    batch_data, batch_label = next(iter(train_loader)) 

    fig = plt.figure()

    for i in range(n_img):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()