from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from torchsummary import summary
import pandas as pd

from utils import GetCorrectPredCount

train_losses = []
test_losses = []
train_acc = []
test_acc = []


def get_layer(layer_type, in_channel=None, out_channel=None,
              dropout_value=0.05, norm='batch', n_group=4,
              ):
    if layer_type == 'C':
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d(out_channel)
        elif norm == 'layer':
            norm_layer = nn.GroupNorm(1, out_channel)
        elif norm == 'group':
            norm_layer = nn.GroupNorm(n_group, out_channel)
        else: 
            raise ValueError('valid inputs for norm are '+
            '(batch, layer, group)')
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            norm_layer,
            nn.Dropout(dropout_value)
        )
    elif layer_type == 'c':
        return nn.Conv2d(in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=(1, 1), padding=0, bias=False)
    elif layer_type == 'P':
        return nn.MaxPool2d(2, 2)
    elif layer_type == 'G':
        return nn.AdaptiveAvgPool2d(output_size=1)

class Net(nn.Module):
    def __init__(self, schema, channels, dropout_value=0.01,
        norm='batch', n_group=4):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        for layer_type, channel_in, channel_out in zip(
            schema, [3, *channels], channels):
            self.layers.append(get_layer(
                layer_type, channel_in, channel_out,
                dropout_value, norm, n_group))

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
    train_losses.append(loss)

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

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

def get_pred_n_actuals(model, test_data, batch_size, device):
    results_df = []
    kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
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
    axs[0, 0].plot([l.cpu().item() for l in train_losses])
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