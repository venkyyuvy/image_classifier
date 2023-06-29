
import torch
from torchvision import datasets
import matplotlib.pyplot as plt


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

def plot_misclassified_images(labels_df, plot_test_data, 
        class_labels,
        nrows=2, ncols=5, figsize=(25,10),
        title='Misclassified images', ):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, )
    fig.suptitle(title, weight='bold', size=14)
    samples = labels_df.query("prediction != target").sample(10)
    axes = axes.ravel()


    for (ix, row), ax in zip(samples.iterrows(), axes):
        img = plot_test_data[ix][0]
        ax.imshow(img)
        ax.set_title(
            f'{class_labels[row.prediction]}|{class_labels[row.target]}',
            fontsize=12)