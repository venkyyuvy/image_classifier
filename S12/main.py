import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_loader import CifarAlbumentationsDataset,\
    CIFAR_CLASS_LABELS, TRAIN_TRANSFORM, TEST_TRANSFORM
from utils import plot_misclassified_images
from explainer import grad_cam
from metrics import get_metrics
import model

import os

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

class LitResnet(LightningModule):
    def __init__(self, lr=0.03, batch_size=512):
        super().__init__()

        self.save_hyperparameters()
        self.model = model.ResNet18()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_probs = self.model(x)
        loss = nn.CrossEntropyLoss()(pred_probs, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        pred_probs = self.model(x)
        loss = nn.CrossEntropyLoss()(pred_probs, y)
        preds = torch.argmax(pred_probs, dim=1)
        acc = accuracy(preds, y, task='multiclass')

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    # todo
    # change the default for num_iter
    def lr_finder(self, optimizer, criterion, 
        num_iter=50, 
    ):
        lr_finder = LRFinder(self, optimizer, criterion,
            device=self.device)
        lr_finder.range_test(
            self.train_dataloader(), end_lr=10,
            num_iter=num_iter, step_mode='exp',
            )
        ax, suggested_lr = lr_finder.plot(suggest_lr=True)
        # todo
        # how to log maplotlib images
        # self.logger.experiment.add_image('lr_finder', plt.gcf(), 0)
        lr_finder.reset() 
        return suggested_lr
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        criterion = nn.CrossEntropyLoss()
        suggested_lr = self.lr_finder(optimizer, criterion)
        steps_per_epoch = len(self.train_dataloader())
        scheduler_dict = {
            "scheduler":  OneCycleLR(
        optimizer, max_lr=suggested_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=self.trainer.max_epochs, 
        pct_start=5/self.trainer.max_epochs,
        three_phase=False,
        div_factor=100,
        final_div_factor=100,
        anneal_strategy='linear',
    ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self, data_path='../data'):
        CifarAlbumentationsDataset(
                data_path, train=True, download=True)
        CifarAlbumentationsDataset(
                data_path, train=False, do0nload=True)

    def setup(self, stage=None, data_dir='../data'):

        if stage == "fit" or stage is None:
            self.train_dataset = CifarAlbumentationsDataset(data_dir, train=True, transform=TRAIN_TRANSFORM)
            self.test_dataset = CifarAlbumentationsDataset(data_dir, train=False, transform=TEST_TRANSFORM)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
        shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
        num_workers=os.cpu_count(), pin_memory=True)


epochs=20
batch_size=512
n_misclassif=20
n_grad_cam=20,



# initialize the trainer
if __name__ == '__main__':
    trainer = Trainer(
        accelerator="mps", devices=1,
        max_epochs = 20,
        enable_progress_bar = True,
    )

    # Train the model
    trainer.fit(LitResnet())
