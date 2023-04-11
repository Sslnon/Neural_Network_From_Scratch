import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import exp


def init_params():
    W1 = np.random.randn(10, 784) - 0.5
    b1 = np.random.randn(10, 1) - 0.5
    W2 = np.random.randn(10, 10) - 0.5
    b2 = np.random.randn(10, 1) - 0.5
    return W1, b1, W2, b2


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
    """ get the one_hot matrix
    :param Y: the label list/array like Y_dev/Y_train
    :return: one_hot matrix of Y
    """
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def deriv_ReLU(Z):
    return Z > 0


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    # get the average diff
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, 1)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, 1)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2


def get_predict(A2):
    return np.argmax(A2, 0)


def get_accuracy(predicts, Y):
    return np.sum(predicts == Y) / Y.size


def gradient_decent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if (i % 50 == 0):
            print("It: ", i)
            print("AC: ", get_accuracy(get_predict(A2),Y))

    return W1, b1, W2, b2


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

    print(X_train.shape)

    W1, b1, W2, b2 = gradient_decent(X_train, Y_train , 100, 0.1)