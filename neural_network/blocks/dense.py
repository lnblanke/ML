# A simple dense layer
# @Time: 10/15/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: dense.py

import numpy as np
from .function import sigmoid, dsigmoid, relu, drelu, softmax, dsoftmax, linear, dlinear
from .layer import Layer


class Dense(Layer):
    def __init__(self, input_size: int, units: int, activation: str, name = None):
        super().__init__(name)
        self.input = None
        self.output = None
        self.units = units
        self.input_size = input_size
        self.weight = np.random.randn(self.units, self.input_size)
        self.bias = np.random.randn(units)

        if activation == "sigmoid":
            self.active_func = sigmoid
            self.active_func_dev = dsigmoid
        elif activation == "relu":
            self.active_func = relu
            self.active_func_dev = drelu
        elif activation == "softmax":
            self.active_func = softmax
            self.active_func_dev = dsoftmax
        else:
            self.active_func = linear
            self.active_func_dev = dlinear

    def feedforward(self, input_vector):
        self.input = input_vector

        return self.active_func(np.dot(self.weight, self.input.flatten()) + self.bias)

    def backprop(self, dy_dx, learning_rate):
        dev = np.dot(dy_dx, self.active_func_dev(np.dot(self.weight, self.input.flatten()) + self.bias)).flatten()
        prop = np.dot(dev, self.weight)

        self.weight -= learning_rate * np.dot(np.transpose(np.asmatrix(dev)), np.asmatrix(self.input.flatten()))
        self.bias -= learning_rate * dev

        return prop.reshape(self.input.shape)
