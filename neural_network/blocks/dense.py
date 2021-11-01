# @Time: 10/15/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: dense.py

import numpy as np
from .function import *

class Dense:
    def __init__(self, input_size, units, activation, learning_rate):
        self.units = units
        self.input_size = input_size
        self.weight = np.random.normal(size = (self.input_size, self.units))
        self.bias = np.random.normal(size = units)
        self.learning_rate = learning_rate

        if activation == "sigmoid":
            self.active_func = sigmoid
            self.active_func_dev = dsigmoid
        elif activation == "relu":
            self.active_func = relu
            self.active_func_dev = drelu

    def feedforward(self, input):
        self.input = input

        self.output = self.active_func(np.dot(np.transpose(self.weight), self.input) + self.bias)

        return self.output

    def backprop(self, dy_dx):
        dev = np.transpose(dy_dx) * np.matrix(self.active_func_dev(self.output)) * np.transpose(self.weight)

        self.weight = self.weight - self.learning_rate * dy_dx * self.active_func_dev(self.output) * self.input
        self.bias -= self.learning_rate * dy_dx * self.active_func_dev(self.output)

        return dev