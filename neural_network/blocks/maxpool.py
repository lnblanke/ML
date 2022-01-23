# A simle MaxPool layer
# @Time: 12/5/21
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: maxpool.py.py
import numpy as np

from .layer import Layer


class MaxPool(Layer):
    def __init__(self, kernel_size: int, name = None):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.input = None

    def feedforward(self, input_vector: np.ndarray):
        h, w, f = input_vector.shape
        self.input = input_vector

        if self.kernel_size > h or self.kernel_size > w:
            raise ArithmeticError("Negative dimension encountered!")

        output = np.empty((h // self.kernel_size, w // self.kernel_size, f))

        for i in range(h // self.kernel_size):
            for j in range(w // self.kernel_size):
                region = input_vector[i * self.kernel_size: i * self.kernel_size + self.kernel_size,
                         j * self.kernel_size: j * self.kernel_size + self.kernel_size]

                output[i, j] = np.max(region, axis = (0, 1))

        return output

    def backprop(self, dy_dx, learning_rate):
        dev = np.zeros(self.input.shape)

        h, w, f = self.input.shape

        for i in range(h // self.kernel_size):
            for j in range(w // self.kernel_size):
                region = self.input[i * self.kernel_size: i * self.kernel_size + self.kernel_size,
                         j * self.kernel_size: j * self.kernel_size + self.kernel_size]

                hh, ww, ff = region.shape
                local_max = np.amax(region, axis = (0, 1))

                for i2 in range(hh):
                    for j2 in range(ww):
                        for k in range(ff):
                            if region[i2, j2, k] == local_max[k]:
                                dev[i * self.kernel_size + i2, j * self.kernel_size + j2, k] = dy_dx[i, j, k]

        return dev
