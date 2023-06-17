Experiment different model architecture for achieving test accuracy more than 99.4% using a model with less than 8k Parameters.

## `Model_1`
### Target:
- Getting the skeleton right
- Perform MaxPooling after the layer with 5 rececptive field
- Include dropout of 10% after all convolutional layers
- Have fully convolutional layer instead of fully connected layer

### Results:
- Parameters: 8.2 K
- Best Train Accuracy: 98.07
- Best Test Accuracy: 98.69

### Analysis:
- Model is underfitting - Current learning rate 0.001 might be too slow; LR scheduler can help
- Adam might perform better than SGD optimizer 
- May be add more channels in initial convolutional layers

## `Model_2`

###  Target:
    - LR scheduler
    - Adam optimizer
    - Fine tune number of channels in nodes

### Results:
- Parameters: 7.9 K
- Best Train Accuracy: 99.01
- Best Test Accuracy: 99.25

### Analsis:
- Model is learning faster now. 
- More channels in the inital layers seems to perform better
- Data augumentation might improve the performance.


## `Model_3`


### Target:
- Data Augumentation

### Results:
- Parameters: 7.9 K
- Best Train Accuracy: 99.01
- Best Test Accuracy: 99.4

### Analysis:
- augumentation helps a lot
