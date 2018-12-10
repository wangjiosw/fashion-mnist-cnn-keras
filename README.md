# CNN - fashion mnist 


## Introduction

Classification of fashion mnist data using convolutional neural networks.

You can modify the model's hyperparameters in *setting.py*.

you also to set training data and testing data path in *setting.py*.

e.g.

- train_data = 'fashionmnist/fashion-mnist_train.csv'
- test_data = 'fashionmnist/fashion-mnist_test.csv'


## Environment:

- python 2.7
- tensorflow
- pandas
- keras

## Run

### test the model in testing data
```
python main.py -test
```
### train the model in training data

```
python main.py -train
```

### run the model in default mode('test')
```
python main.py
```

## Result

After 20 iterations, the final accuracy of the test data with 10k samples was 91.82%.