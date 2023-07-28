import utils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from data_loader import prepare_cifar_dataloader,\
    CIFAR_CLASS_LABELS
from utils import plot_misclassified_images
from torchsummary import summary
import model

## params
device = 'mps:0'
epochs = 20
batch_size = 126

train_loader, test_loader = prepare_cifar_dataloader()
utils.plot_img_batch(
    train_loader, CIFAR_CLASS_LABELS,
    ncols=15, nrows=4)


network = model.ResNet18()
print(model.summary_printer(network))
network.to(device)

optimizer = optim.SGD(network.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()
suggested_lr = model.lr_finder(
    network, optimizer, criterion, device, train_loader)


scheduler = OneCycleLR(
    optimizer, max_lr=suggested_lr,
    steps_per_epoch=len(train_loader),
    epochs=epochs, 
    pct_start=5/epochs, 
    three_phase=False,
    div_factor=100,
    final_div_factor=100,
    anneal_strategy='linear',
)

for epoch in range(epochs):
    print("EPOCH:", epoch)
    model.train(network, device, train_loader, optimizer, scheduler)
    test_acc = model.test(network, device, test_loader)

model.plot_loss_n_acc()
labels_df = model.get_pred_n_actuals(network, test_loader, device)
plot_misclassified_images(labels_df, test_loader.dataset, 
        CIFAR_CLASS_LABELS, n_samples=20,
        nrows=2, ncols=10, figsize=(25,6),
        title='misclassified images (Pred | Target)', )

from metrics import get_metrics
get_metrics(labels_df, CIFAR_CLASS_LABELS)