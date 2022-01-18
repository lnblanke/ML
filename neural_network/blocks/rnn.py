# A simple RNN layer
# @Time: 12/24/21
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: rnn.py.py
import numpy as np


class RNN(Layer):
    def __init__(self, input_size: int, output_size: int, units: int, name = None, output_seq = False):
        super().__init__(name)

        if output_seq:
            raise NotImplementedError()

        self.input_size = input_size
        self.output_size = output_size
        self.units = units
        self.output_seq = output_seq
        self.weight_xh = np.random.randn(self.units, self.input_size)
        self.weight_hh = np.random.randn(self.units, self.units)
        self.weight_hy = np.random.randn(self.output_size, self.units)
        self.bias_h = np.zeros(self.units)
        self.bias_y = np.zeros(self.output_size)
        self.seq = None
        self.hs = None
        self.input = None
        self.length = 0

    def feedforward(self, input_vector: np.ndarray):
        h = np.zeros(self.units)
        y = np.zeros(self.output_size)
        self.seq = []
        self.hs = []
        self.length = len(input_vector)
        self.input = input_vector

        for i in range(self.length):
            h = np.tanh(np.dot(self.weight_xh, input_vector[i]) + np.dot(self.weight_hh, h) + self.bias_h)
            y = np.dot(self.weight_hy, h) + self.bias_y
            self.seq.append(y)
            self.hs.append(h)

        if self.output_seq:
            return self.seq
        else:
            return y

    def backprop(self, dy_dx: np.ndarray, learning_rate):
        self.weight_hy -= learning_rate * np.dot(np.asmatrix(dy_dx).transpose(), np.asmatrix(self.hs[-1]))
        self.bias_y -= learning_rate * dy_dx
        d_hh = np.zeros(self.weight_hh.shape)
        d_xh = np.zeros(self.weight_xh.shape)
        d_hb = np.zeros(self.bias_h.shape)
        d_h = np.dot(dy_dx, self.weight_hy)

        for i in reversed(range(self.length)):
            dev = (1 - self.hs[i - 1] ** 2) * d_h
            d_hh += np.dot(np.asmatrix(dev).transpose(), np.asmatrix(self.hs[i - 1]))
            d_xh += np.dot(np.asmatrix(dev).transpose(), np.asmatrix(self.input[i]))
            d_hb += dev
            d_h = np.dot(self.weight_hh, dev)

        self.weight_hh -= learning_rate * d_hh
        self.weight_xh -= learning_rate * d_xh
        self.bias_h -= learning_rate * d_hb
