from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from data_loader import NORM_DATA_MEAN, NORM_DATA_STD

def denormalize(data, mean=NORM_DATA_MEAN, std=NORM_DATA_STD):
    return data * std + mean

def grad_cam(model, target_layers, input_tensor, true_class,
    use_cuda=False):
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    targets = [ClassifierOutputTarget(true_class)]

    # you can also pass aug_smooth=true and eigen_smooth=true, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    rgb_imgs = input_tensor.cpu().numpy()
    print(rgb_imgs.shape)
    return [show_cam_on_image(
        denormalize(rgb_img),  #[0].transpose(1, 2, 0)
        cam_output, use_rgb=True, image_weight=0.7)
    for rgb_img, cam_output in zip(rgb_imgs, grayscale_cam)]