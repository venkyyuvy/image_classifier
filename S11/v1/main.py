import utils
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from data_loader import prepare_cifar_dataloader,\
    CIFAR_CLASS_LABELS
from utils import plot_misclassified_images
from explainer import grad_cam
from metrics import get_metrics
import model

## params
def run_experiment(
    device='mps:0',
    epochs=20,
    batch_size=512,
    n_misclassif=20,
    n_grad_cam=20, 
    writer=SummaryWriter('GradCam'),
):
    train_loader, test_loader = prepare_cifar_dataloader(
        batch_size=batch_size
    )
    utils.plot_img_batch(
        train_loader, CIFAR_CLASS_LABELS,
        ncols=10, nrows=4)

    network = model.ResNet18()
    print(model.summary_printer(network))
    network.to(device)


    optimizer = optim.SGD(network.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    suggested_lr = model.lr_finder(
        network, optimizer, criterion, device, train_loader,
        writer)

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
        model.train(network, device, 
            train_loader, optimizer, scheduler, writer)
        test_acc = model.test(network, device, test_loader, writer)

    model.plot_loss_n_acc(writer)

    ## plot_misclassified images
    labels_df = model.get_pred_n_actuals(network, test_loader, device)
    plot_misclassified_images(labels_df, test_loader.dataset, 
            CIFAR_CLASS_LABELS, n_samples=n_misclassif,
            nrows=2, ncols=10, figsize=(25,6),
            title='misclassified images (Pred | Target)', )

    get_metrics(labels_df, CIFAR_CLASS_LABELS)


    ## Grad cam visualization
    target_layers = [network.layer3[-1]]
    inputs, target = next(iter(train_loader))
    input_tensor = inputs[:n_grad_cam].to(device)
    true_class = target[:n_grad_cam].to(device)
    cam_output = grad_cam(network, target_layers,
        input_tensor, None)