'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from collections import OrderedDict
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from torchsummary import summary
import pandas as pd
from torch_lr_finder import LRFinder

from utils import GetCorrectPredCount
from torch.utils.tensorboard import SummaryWriter

train_losses = []
test_losses = []
train_acc = []
test_acc = []


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def train(model, device, train_loader, optimizer, scheduler):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  criterion = nn.CrossEntropyLoss()
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_losses.append(loss.item())

    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Update pbar-tqdm

    correct += GetCorrectPredCount(y_pred, target)
    processed += len(data)

    pbar.set_description(
        desc= f'Loss={loss.item():0.4f} Batch_id={batch_idx} '+
        f'LR={scheduler.get_last_lr()[0]:0.5f} '+
        f'Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    processed = 0
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(test_loader)
    with torch.no_grad():
      for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            correct += GetCorrectPredCount(output, target)
            processed += len(data)



    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))

    test_acc.append(acc)
    return acc

def get_pred_n_actuals(model, test_loader, device):
    results_df = []
    for _, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred_label = output.argmax(dim=1)
        results_df.append(
            pd.DataFrame({"prediction": pred_label.cpu().numpy(),
                          "target": target.cpu().numpy()}))

    return pd.concat(results_df).reset_index(drop=True)

def plot_loss_n_acc():
    """
    plots the train and test
    losses and accuracies
    Parameters
    ----------
    losses : (train_losses, test_losses)
        train and test losses values for each batch
    accuracies : (train_acc, test_acc)
        train and test accuracy values for each batch
    """
    # performance and loss curves
    fig, axs = plt.subplots(2, 2, figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def summary_printer(model, device='cpu'):
    model = model.to(device)
    return summary(model, input_size=(3, 28, 28))

def lr_finder(
    model, optimizer, criterion, device, train_loader,
    num_iter=100,
    ):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader, end_lr=10, num_iter=num_iter, step_mode='exp',
        )
    _, suggested_lr = lr_finder.plot(suggest_lr=True)
    lr_finder.reset() 
    return suggested_lr