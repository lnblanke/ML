# A simple activation layer
# @Time: 1/25/22
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Activation.py.py

# A simple dense layer
# @Time: 10/15/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: dense.py

import numpy as np
from .function import sigmoid, dsigmoid, relu, drelu, softmax, dsoftmax, linear, dlinear
from .layer import Layer


class Activation(Layer):
    def __init__(self, units: int, activation: str, name = None):
        super().__init__(name)
        self.units = units
        self.input = None

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
        return self.active_func(input_vector)

    def backprop(self, dy_dx, learning_rate):
        dev = np.dot(self.active_func_dev(self.input), dy_dx)

        return dev.reshape(self.input.shape)
