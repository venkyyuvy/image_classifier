## Dataset preparation:

### CIFAR 10 dataset
### Data augumentation
## Model summary:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
              ReLU-2           [-1, 64, 32, 32]               0
         GroupNorm-3           [-1, 64, 32, 32]             128
           Dropout-4           [-1, 64, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]          18,432
              ReLU-6           [-1, 32, 32, 32]               0
         GroupNorm-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 16, 32, 32]             512
        MaxPool2d-10           [-1, 16, 16, 16]               0
           Conv2d-11            [-1, 8, 16, 16]           1,152
             ReLU-12            [-1, 8, 16, 16]               0
        GroupNorm-13            [-1, 8, 16, 16]              16
          Dropout-14            [-1, 8, 16, 16]               0
           Conv2d-15           [-1, 16, 16, 16]           1,152
             ReLU-16           [-1, 16, 16, 16]               0
        GroupNorm-17           [-1, 16, 16, 16]              32
          Dropout-18           [-1, 16, 16, 16]               0
           Conv2d-19           [-1, 16, 16, 16]           2,304
             ReLU-20           [-1, 16, 16, 16]               0
        GroupNorm-21           [-1, 16, 16, 16]              32
          Dropout-22           [-1, 16, 16, 16]               0
           Conv2d-23           [-1, 32, 16, 16]             512
        MaxPool2d-24             [-1, 32, 8, 8]               0
           Conv2d-25              [-1, 8, 8, 8]           2,304
             ReLU-26              [-1, 8, 8, 8]               0
        GroupNorm-27              [-1, 8, 8, 8]              16
          Dropout-28              [-1, 8, 8, 8]               0
           Conv2d-29             [-1, 16, 8, 8]           1,152
             ReLU-30             [-1, 16, 8, 8]               0
        GroupNorm-31             [-1, 16, 8, 8]              32
          Dropout-32             [-1, 16, 8, 8]               0
           Conv2d-33             [-1, 32, 8, 8]           4,608
             ReLU-34             [-1, 32, 8, 8]               0
        GroupNorm-35             [-1, 32, 8, 8]              64
          Dropout-36             [-1, 32, 8, 8]               0
AdaptiveAvgPool2d-37             [-1, 32, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             320
================================================================
Total params: 34,560
Trainable params: 34,560
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.66
Params size (MB): 0.13
Estimated Total Size (MB): 3.80
----------------------------------------------------------------


## Batch Normalization
Training accuracy: 74.35%
Test accuracy: 71.53%
### Performance curves:
### Misclassification plots

## Layer Normalization
Training accuracy:  
Test accuracy:
### Performance curves:
### Misclassification plots

## Group Normalization
Training accuracy:  
Test accuracy:
### Performance curves:
### Misclassification plots
