
# Grad CAM explanability for ResNet 18 architecture 

## dataset

CIFAR10 classification

## main.py
- uses OneCyclePolicy for learning rate schedule (internally uses `lr_finder`)
- trains resnet-18 for 20 epochs
- achieves 90% test accuracy

## explainer.py

Applies the Grad class activation mapping explainer on last conv layer of the third block