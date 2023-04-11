import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import exp


def init_params():
    W1 = np.random.randn(10, 784) - 0.5
    b1 = np.random.randn(10, 1) - 0.5
    W2 = np.random.randn(10, 10) - 0.5
    b2 = np.random.randn(10, 1) - 0.5


def ReLU(Z):
    return np.maximum(0, Z)


def softmax(Z):
    return exp(Z) / np.sum(exp(Z))


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size),Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def back_prop(Z1, A1, Z2, A2, Y):
    one_hot_Y = one_hot(Y)



if __name__ == '__main__':
    data = pd.read_csv("mnist_train.csv")

    # print(data.head())

    m, n = data.shape

    data = np.array(data)
    np.random.shuffle(data)

    data_dev = data[0:1000].T
    Y_dev = data[0]
    X_dev = data[1:n]

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
