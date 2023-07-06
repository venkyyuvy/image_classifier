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


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels,
        dropout_value):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                    kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_value)
            )
        
    def forward(self, x):
        return self.layer(x)




class ConvLayer(nn.Module):
    def __init__(self, in_channel=None, out_channel=None,
              dropout_value=0.05, stride=1,
              kernel_size=3, padding=1, dilation=1,
              ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(dropout_value)
        )

    def forward(self, x):
        return self.layer(x)

class Net(nn.Module):
    def __init__(self, p=0.01):
        super(Net, self).__init__()
        self.layers = nn.ModuleDict({
            'layer0' : ConvLayer(3, 64, dropout_value=p),
            'layer1_a' : nn.Sequential(
                OrderedDict({
                    'conv1': nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    'mp1': nn.MaxPool2d(2),
                    'bn': nn.BatchNorm2d(128),
                    'act': nn.ReLU(),
                    'dp': nn.Dropout(p=p)
                })
            ),
            'layer1_b': nn.Sequential(
                OrderedDict({
                'layer1_conv1': ConvLayer(128, 128, dropout_value=p),
                'layer1_conv2': ConvLayer(128, 128, dropout_value=p),
                })
            ),
            'layer2': nn.Sequential(
                OrderedDict({
                    'r1_conv1': nn.Conv2d(128, 256, 3, padding=1, bias=False),
                    'r1_mp1': nn.MaxPool2d(2),
                    'r1_bn': nn.BatchNorm2d(256),
                    'r1_act': nn.ReLU(),
                    'r1_dp': nn.Dropout(p=p)
                })
            ),
            'layer3_a' : nn.Sequential(
                OrderedDict({
                    'r2_conv1': nn.Conv2d(256, 512, 3, padding=1, bias=False),
                    'r2_mp1': nn.MaxPool2d(2),
                    'r2_bn': nn.BatchNorm2d(512),
                    'r2_act': nn.ReLU(),
                    'r2_dp': nn.Dropout(p=p)
                })
            ),
            'layer3_b': nn.Sequential(
                OrderedDict({
                'layer3_conv1': ConvLayer(512, 64, dropout_value=p), 
                'layer3_conv2': ConvLayer(64, 512, dropout_value=p),
                })
            ),
            'final_mp': nn.MaxPool2d(4),
            'fc': nn.Linear(512, 10, bias=False),
        })


    def forward(self, x):
        for name, layer in self.layers.items():
            if name == 'layer1_b' or name == 'layer3_b':
                x = x + layer(x)
                continue
            elif name == 'fc':
                x = x.view(-1, 512)
            x = layer(x)
        return F.softmax(x, dim=-1)

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
            pbar.set_description(
            desc= f'Test Loss={test_loss:0.4f} Batch_id={batch_idx} '+
            f'Accuracy={100*correct/processed:0.2f}')


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

def summary_printer(model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    return summary(model, input_size=(1, 28, 28))

def lr_finder(model, optimizer, criterion, device, train_loader):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, 
                         end_lr=1, num_iter=200, step_mode='exp')
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state