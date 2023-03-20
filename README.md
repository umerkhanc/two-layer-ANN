# A Neural Network with One Hidden Layer

In this project, a neural network ith one hidden layer is implemented from scratch in Python. The model is trained using stocastic gradient descent (SGD) and evaluated on the MNIST dataset.

## Dependencies

```
numpy==1.16.4
h5py==2.9.0
matplotlib==3.1.0
```

## Dataset

The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset is used to train and evaluate the neural network model in this project. It is a database of handwritten digits that is commonly used to train image processing models. The dataset in hdf5 format is included in the repository.

## Implementation

#### Activation functions

Two types of activation functions are included: tanh and sigmoid functions. The evaluation of the function and the derivative of the function are implemented as follow.

```python
def sigma(z, func):
    """
    Activation functions

    Parameters
    ----------
    z : ndarray of float
        input
    func : str
        the type of activation adopted

    Returns
    -------
    ZZ : ndarray of float
        output
    """
    if func == 'tanh':
        ZZ = np.tanh(z)
    elif func == 'sigmoid':
        ZZ = np.exp(z)/(1 + np.exp(z))
    else:
        sys.exit("Unsupported function type!")
    return ZZ
```
```python
def d_sigma(z, func):
    """
    Derivative of activation functions

    Parameters
    ----------
    z : ndarray of float
        input
    func : str
        the type of activation

    Returns
    -------
    dZZ : ndarray of float
        output
    """
    if func == 'tanh':
        dZZ = 1.0 - np.tanh(z)**2
    elif func == 'sigmoid':
        dZZ = np.exp(z)/(1 + np.exp(z)) * (1 - np.exp(z)/(1 + np.exp(z)))
    else:
        sys.exit("Unsupported function type!")
    return dZZ
```
#### Softmax function

The softmax function is applied in the output layer of the neural network.
```python
def softmax_function(z):
    """
    Softmax function

    Parameters
    ----------
    z : ndarray of float
        input

    Returns
    -------
    ZZ : ndarray of float
        output
    """
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ
```

#### Forward propagation
```python
def forward(x, model, func):
    """
    Forward propagation of the neural network

    Parameters
    ----------
    x : ndarray of float
        input
    model : dict
        parameters/weights of the nerual network
    func : str
        the type of activation

    Returns
    -------
    Z : ndarray of float
        output of the linear layer
    H : ndarray of float
        output after the activation
    f : ndarray of float
        output of the forward propagation
    """
    Z = np.dot(model['W'], x) + model['b1']
    H = sigma(Z,func)
    U = np.dot(model['C'], H) + model['b2']
    f = softmax_function(U)
    return (Z, H, f)
```

#### Backpropagation
```python
def backprop(x, y, f, Z, H, model, model_grads, func):
    """
    Backpropagation of the neural network

    Parameters
    ----------
    x : ndarray of float
        input
    y : ndarray of int
        ground truth label
    f : ndarray of float
        output of the forward propagation
    Z : ndarray of float
        output of the linear layer
    H : ndarray of float
        output after the activation
    model : dict
        parameters/weights of the nerual network
    model_grads : dict
        gradients of the parameters/weights of the nerual network
    func : str
        the type of activation

    Returns
    -------
    model_grads : dict
        updated gradients of the parameters/weights of the nerual network
    """
    dU = - 1.0*f
    dU[y] = dU[y] + 1.0
    db2 = dU
    dC = np.outer(dU, np.transpose(H))
    delta = np.dot(np.transpose(model['C']), dU)
    dZZ = d_sigma(Z,func)
    db1 = np.multiply(delta, dZZ)
    dW = np.outer(db1,np.transpose(x))
    model_grads['W'] = dW
    model_grads['C'] = dC
    model_grads['b1'] = db1
    model_grads['b2'] = db2
    return model_grads
```

## Hyerparameters

The neural network is trained using stochastic gradient descent (SGD) algorithm. Staircase decaying strategy is used for learning rate scheduling.

The default hyperparameters are set as follow:

| Hyperparameter               | Value   |
| ---------------------------- |:-------:|
| Initial learning rate        | 0.1     |
| Learning rate decay factor   | 0.1     |
| Learning rate decay interval | 5       |
| Number of epochs             | 20      |
| Number of hidden units       | 64      |
| Activation function (sigma)  | sigmoid |


## Running the model

Run the following script to make sure required packages are installed.
```
pip install -r requirements.txt
```
To run a quick test of the pipeline with a small network (8 hidden units) and a small number of training steps (1 epoch)
```
python main.py --quicktest
```
To run the model with the default hyperparameters
```
python main.py
```
To run the model with specified hyperparameters
```
python main.py --lr my_lr --decay my_decay --interval my_interval --n_epochs my_n_epochs --n_h my_n_h --sigma my_sigma
```
To see all available options and their descriptions
```
python main.py --help
```

## Result

With the default hyperparameters, the training and evaluation results are as follow.

```
Hyperparameters
-----------------
Initial learning rate : 0.1000
Learning rate decay : 0.1000
Staircase learning rate decay interval : 5
Number of epochs : 20
Number of hidden units : 64
Activation function : sigmoid

MNIST data info
----------------
Number of training data : 60000
Number of test data : 10000
Input data shape : 784
Output data shape : 10

Start training
---------------
Epoch   0,  Accuracy 0.9305
Epoch   1,  Accuracy 0.9635
Epoch   2,  Accuracy 0.9721
Epoch   3,  Accuracy 0.9751
Epoch   4,  Accuracy 0.9770
Epoch   5,  Accuracy 0.9863
Epoch   6,  Accuracy 0.9894
Epoch   7,  Accuracy 0.9908
Epoch   8,  Accuracy 0.9913
Epoch   9,  Accuracy 0.9926
Epoch  10,  Accuracy 0.9928
Epoch  11,  Accuracy 0.9927
Epoch  12,  Accuracy 0.9933
Epoch  13,  Accuracy 0.9928
Epoch  14,  Accuracy 0.9929
Epoch  15,  Accuracy 0.9931
Epoch  16,  Accuracy 0.9930
Epoch  17,  Accuracy 0.9927
Epoch  18,  Accuracy 0.9936
Epoch  19,  Accuracy 0.9936
Training Time : 352.5622 (s)

Start testing
--------------
Test Accuracy : 0.9753
```

Examples of correctly predicted images

![img_correct_1](./figs/correct_1.png)
![img_correct_2](./figs/correct_2.png)
![img_correct_3](./figs/correct_3.png)
![img_correct_4](./figs/correct_4.png)
![img_correct_5](./figs/correct_5.png)

Examples of *incorrectly* predicted images

![img_incorrect_1](./figs/incorrect_1.png)
![img_incorrect_2](./figs/incorrect_2.png)
![img_incorrect_3](./figs/incorrect_3.png)
![img_incorrect_4](./figs/incorrect_4.png)
![img_incorrect_5](./figs/incorrect_5.png)
