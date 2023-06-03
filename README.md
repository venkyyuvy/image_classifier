# Image Classifier

This repository contains the scripts to develop a image classifier in PyTorch.

# Usage

> ## `Utils.py`
> Plotting contains functions for primarily two process.
> ### Data preparation
> This python script provides the neccessary functions for preparing the dataset by doing the following operations
> - Builds a PyTorch Dataset using downloaded MNIST dataset 
> - DataLoader which includes data transformation
>
> ### Plotting
> It also provides function for plotting a batch of images and plotting the performance curves.
> ## `Model.ipynb`
> This script helps us create a PyTorch convolution model with following architecture and it has the functions for training and testing the script.
>
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 32, 26, 26]             320
                Conv2d-2           [-1, 64, 24, 24]          18,496
                Conv2d-3          [-1, 128, 10, 10]          73,856
                Conv2d-4            [-1, 256, 8, 8]         295,168
                Linear-5                   [-1, 50]         204,850
                Linear-6                   [-1, 10]             510
    ================================================================
    Total params: 593,200
    Trainable params: 593,200
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.67
    Params size (MB): 2.26
    Estimated Total Size (MB): 2.94
    ----------------------------------------------------------------
> ## `S5.ipynb`
> This notebooks high level scripts to build the overall pipeline tofor building the Image classifier and render visualizations of images and model performance.