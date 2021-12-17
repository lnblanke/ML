# Functions that are used in machine learning
# @Time: 9/23/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: function.py

import numpy as np


# Mean square error
# MSE = (predict value-real value)^2/n
def mse(pred, real):
    return np.sum((pred - real) ** 2) / len(pred)


# Derivative of mean square error
# MSE' = (predict value-real value)/n
def dmse(pred, real):
    return (pred - real) / len(pred)


# Cross entropy loss
def cross_entropy_loss(pred, real):
    return -1 * np.log(pred[real])


# Derivative of cross entropy loss
def dcross_entropy_loss(pred, real):
    grad = np.zeros(pred.shape)
    grad[real] = -1 / pred[real]

    return grad


# Sigmoid function
# Sigmoid(x) = 1/(1+e^-x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Deriative of Sigmoid
# Sigmoid' (x) = Sigmoid (x) * (1-Sigmoid (x))
def dsigmoid(x):
    dev = np.zeros((len(x), len(x)))
    np.fill_diagonal(dev, sigmoid(x) * (1 - sigmoid(x)))
    return dev


# ReLU function
# ReLU(x) = max(0, x)
def relu(x):
    return np.maximum(0, x)


# Derivative of ReLU
# Relu'(x) = 1 if x >= 0 otherwise 0
def drelu(x):
    dev = np.zeros((len(x), len(x)))
    np.fill_diagonal(dev, np.maximum(0, x / abs(x)))
    return dev


# Softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


# Derivative of Softmax function
def dsoftmax(x):
    dev = np.zeros((len(x), len(x)))

    for i in range(len(x)):
        dev[:, i] = - np.exp(x[i]) * np.exp(x) / np.sum(np.exp(x)) ** 2

    np.fill_diagonal(dev, np.exp(x) * (np.sum(np.exp(x)) - np.exp(x)) / np.sum(np.exp(x)) ** 2)

    return dev


# Linear activation
def linear(x):
    return x


# Derivative of linear activation
def dlinear(x):
    return 1
