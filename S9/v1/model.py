from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from torchsummary import summary
import pandas as pd

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
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_value)
            )
        
    def forward(self, x):
        return self.layer(x)


def get_start_layer(out_channel, dropout_value):
    return nn.Sequential(
        nn.Conv2d(in_channels=3,
                    out_channels=out_channel,
                    kernel_size=(7, 7), padding=3, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(out_channel),
        nn.Dropout(dropout_value)
    )

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
    def __init__(self, channels, dropout_value=0.01):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            # C1 block
            get_start_layer(channels[0], dropout_value),
            ConvLayer(channels[0], channels[1], dropout_value=dropout_value),
            ConvLayer(channels[1], channels[2], dropout_value=dropout_value),
            # strided convolution
            ConvLayer(channels[2], channels[3],
                dropout_value=dropout_value, stride=2),

            # C2 BLOCK
            nn.Conv2d(channels[3], channels[4], kernel_size=1,
                bias=False),
            ConvLayer(channels[4], channels[5], dropout_value=dropout_value),
            ConvLayer(channels[5], channels[6], dropout_value=dropout_value,
                dilation=2), 
            ConvLayer(channels[6], channels[7], dropout_value=dropout_value,
                stride=2),

            # C3 BLOCK
            nn.Conv2d(channels[7], channels[8], kernel_size=1,
                bias=False),
            ConvLayer(channels[8], channels[9], dropout_value=dropout_value),
            DepthwiseSeparableConv(channels[9], channels[10], 
                dropout_value=dropout_value,),
            ConvLayer(channels[10], channels[11], dropout_value=dropout_value,
                stride=2),
            
            # C4 BLOCK
            nn.Conv2d(channels[11], channels[12], kernel_size=1,
                bias=False),
            ConvLayer(channels[12], channels[13], dropout_value=dropout_value),
            ConvLayer(channels[13], channels[14], dropout_value=dropout_value,
                dilation=2),
            ConvLayer(channels[14], channels[15], dropout_value=dropout_value,
                stride=2),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(channels[16], channels[17], kernel_size=1,
                bias=False), 
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

def train(model, device, train_loader, optimizer):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss.cpu().item())

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    correct += GetCorrectPredCount(y_pred, target)
    processed += len(data)

    pbar.set_description(
        desc= f'Loss={loss.item():0.4f} Batch_id={batch_idx} '+
        f'Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            correct += GetCorrectPredCount(output, target)


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