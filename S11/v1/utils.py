import numpy as np
import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from data_loader import NORM_DATA_MEAN, NORM_DATA_STD


def prepare_mnist_data(train_transforms, test_transforms,
    data_path='../data', batch_size=512):
    train_data = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transforms)
    test_data = datasets.CIFAR10(data_path, train=False, download=True, transform=test_transforms)

    kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    return train_loader, test_loader

def denormalize(data, mean=np.array(NORM_DATA_MEAN), 
                std=np.array(NORM_DATA_STD)):
    return data * std + mean

def plot_img_batch(train_loader, class_labels, nrows=2, ncols=6):
    batch_data, batch_label = next(iter(train_loader)) 

    fig = plt.figure(figsize=(ncols, nrows))
    n_img = nrows * ncols
    for i in range(n_img):
        plt.subplot(nrows, ncols, i+1)
        plt.tight_layout()
        data = denormalize(np.transpose(batch_data[i], (1, 2, 0)))
        data = data.numpy().clip(0, 255)
        plt.imshow(data)
        plt.title(class_labels[batch_label[i].item()], fontsize=5)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.subplots_adjust(hspace=0.3, wspace=0.1)


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def plot_misclassified_images(labels_df, test_dataset, 
        class_labels, n_samples=10,
        nrows=2, ncols=5, figsize=(25,10),
        title='misclassified images', ):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, )
    fig.suptitle(title, weight='bold', size=14)
    samples = labels_df.query("prediction != target").sample(n_samples)
    axes = axes.ravel()
    subset_indices = samples.index.values
    subset = torch.utils.data.Subset(test_dataset, subset_indices)
    for i, ((ix, row), ax) in enumerate(zip(samples.iterrows(), axes)):
        data = subset[i][0]
        data = denormalize(np.transpose(data, (1, 2, 0)))
        img = data.numpy().clip(0, 255)
        ax.imshow(img)
        ax.set_title(
            f'{class_labels[row.prediction]}|{class_labels[row.target]}',
            fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')