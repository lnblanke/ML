# A simple convolutional layer
# @Time: 12/5/21
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: conv.py.py

from .layer import Layer
import numpy as np


class Conv(Layer):
    def __init__(self, kernal_size: int, filters: int, padding: str, name = None):
        super().__init__(name)

        self.kernel_size = kernal_size
        self.filters = filters
        self.padding = padding
        self.weights = np.random.randn(filters, kernal_size, kernal_size)
        self.input = None

        if padding != "same" and padding != "valid":
            raise NameError("The name of padding is not found!")

    def feedforward(self, input_vector: np.ndarray):
        h, w = input_vector.shape
        self.input = input_vector

        if self.kernel_size > h or self.kernel_size > w:
            raise ArithmeticError("Negative dimension encountered!")

        if self.padding == "same":
            output = np.zeros((h, w, self.filters))
        else:
            output = np.empty((h - self.kernel_size + 1, w - self.kernel_size + 1, self.filters))

            for i in range(h - self.kernel_size + 1):
                for j in range(w - self.kernel_size + 1):
                    region = input_vector[i: i + self.kernel_size, j: j + self.kernel_size]
                    output[i, j] = np.sum(region * self.weights, axis = (1, 2))

        return output

    def backprop(self, dy_dx: np.ndarray, learning_rate):
        dev = np.zeros((self.filters, self.kernel_size, self.kernel_size))

        h, w = self.input.shape

        for i in range(h - self.kernel_size + 1):
            for j in range(w - self.kernel_size + 1):
                region = self.input[i: i + self.kernel_size, j: j + self.kernel_size]
                for k in range(self.filters):
                    dev[k] += dy_dx[i, j, k] * region

        self.weights -= learning_rate * dev

        # TODO: add return derivative for next bp
