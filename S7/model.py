from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from matplotlib import pyplot as plt
from torchsummary import summary

from utils import GetCorrectPredCount

train_losses = []
test_losses = []
train_acc = []
test_acc = []


class Net(nn.Module):
    def __init__(self, drop_p=0.15):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(drop_p)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(drop_p)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 128, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(drop_p)
        )
        self.avg = nn.AvgPool2d(2)
        self.fc = nn.Sequential(
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

def model_train(model, device, train_loader, optimizer):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0


    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = F.nll_loss(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(
            desc= f'Train: Loss={loss.item():0.4f} \
            Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))


def model_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss


            correct += GetCorrectPredCount(output, target)

        test_loss /= len(test_loader.dataset)
        test_acc.append(100. * correct / len(test_loader.dataset))
        test_losses.append(test_loss)
    
    print('Test set: Average loss: {:.4f},\
        Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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