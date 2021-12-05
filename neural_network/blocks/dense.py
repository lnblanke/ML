# @Time: 10/15/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: dense.py
import numpy as np

from .function import *


class Dense:
    def __init__(self, input_size, units, activation, learning_rate):
        self.input = None
        self.output = None
        self.units = units
        self.input_size = input_size
        self.weight = np.random.normal(size = (self.units, self.input_size))
        self.bias = np.random.normal(size = (units))
        self.learning_rate = learning_rate

        if activation == "sigmoid":
            self.active_func = sigmoid
            self.active_func_dev = dsigmoid
        elif activation == "relu":
            self.active_func = relu
            self.active_func_dev = drelu

    def feedforward(self, input_vector):
        self.input = input_vector
        self.output = np.empty(shape = (len(input_vector), self.units))

        for i in range(len(input_vector)):
            self.output[i] = self.active_func(np.dot(self.weight, self.input[i]) + self.bias)

        return self.output

    def backprop(self, dy_dx):
        prop = np.empty(shape = (len(dy_dx), self.input_size))

        for i in range(len(dy_dx)):
            dev = np.multiply(dy_dx[i], self.active_func_dev(np.dot(self.weight, self.input[i]) + self.bias))

            prop[i] = np.dot(dev, self.weight)

            self.weight -= self.learning_rate * np.dot(np.transpose(np.asmatrix(dev)), np.asmatrix(self.input[i]))
            self.bias -= self.learning_rate * dev

        return prop
