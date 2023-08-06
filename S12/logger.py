import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

class TensorFlowLoggerCallback(pl.Callback):
    def __init__(self):
        super(TensorFlowLoggerCallback, self).__init__()
        self.tf_writer = None

    def on_train_start(self, trainer, pl_module):
        # Initialize the TensorFlow SummaryWriter when training starts
        self.tf_writer = SummaryWriter(log_dir=trainer.log_dir)

    def on_train_epoch_end(self, trainer, pl_module):
        # Log metrics to TensorFlow at the end of each epoch
        metrics = trainer.callback_metrics
        global_step = trainer.global_step
        for key, value in metrics.items():
            self.tf_writer.add_scalar(key, value, global_step=global_step)

    def on_train_end(self, trainer, pl_module):
        # Close the SummaryWriter when training ends
        self.tf_writer.close()