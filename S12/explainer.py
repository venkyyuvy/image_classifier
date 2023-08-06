import numpy as np
import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from data_loader import NORM_DATA_MEAN, NORM_DATA_STD

def denormalize(data, mean=NORM_DATA_MEAN, std=NORM_DATA_STD):
    return data * std + mean

def grad_cam(model, target_layers, input_tensor, target_class,
    use_cuda=False):
  
    cam = GradCAM(model=model, target_layers=target_layers,
        use_cuda=use_cuda)

    if target_class is not None:
        target = [ClassifierOutputTarget(target_class)]
    else:
        target = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=target)
    results = []
    for img_tensor, cam_output in zip(input_tensor, grayscale_cam):
        rgb_img = img_tensor.cpu().numpy()
        results.append(show_cam_on_image(
            denormalize(rgb_img.transpose(1, 2, 0)),
            cam_output, use_rgb=True, image_weight=0.7))

    cam_output = torch.tensor(np.array(results), dtype=float).permute(0, 3, 1, 2)
    grid = make_grid(cam_output,
        nrow=10, padding=2, normalize=True)

    # Convert the grid to a NumPy array
    grid_np = grid.numpy().transpose((1, 2, 0))

    # Plot the grid
    plt.imshow(grid_np)
    plt.axis('off')
    plt.show()