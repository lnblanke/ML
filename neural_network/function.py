# Functions that are used in machine learning
# @Time: 9/23/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: function.py

import numpy as np

# Sigmoid Function
# Sigmoid (x) = 1/(1+e^-x)
def sigmoid(x):
    return 1 / (1 + np.exp(- x))

# Deriative of sigmoid
# Sigmoid' (x) = Sigmoid (x) * (1-Sigmoid (x))
def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.max(0, x)

# Mean Square Loss
# MSE = (predict value-real value)^2/n
def mse(pred, real):
    return ((pred - real) ** 2).mean()