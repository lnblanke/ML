# Functions that are used in machine learning
# @Time: 9/23/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: function.py

import numpy as np

# Sigmoid function
# Sigmoid (x) = 1/(1+e^-x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Deriative of Sigmoid
# Sigmoid' (x) = Sigmoid (x) * (1-Sigmoid (x))
def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# ReLU function
def relu(x):
    return np.max(0, x)

# Derivative of ReLU
# Relu'(x) = 1 if x >= 0
def drelu(x):
    return 1 if x >= 0 else 0

# Mean Square Loss
# MSE = (predict value-real value)^2/n
def mse(pred, real):
    return ((pred - real) ** 2).mean()
