# @Time: 8/14/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: rnn_class.py

import numpy as np

class recur:
    def __init__(self, input_size, output_size, hidden_layer = 64):
        self.Whh = np.random.randn(hidden_layer, hidden_layer) / 1000
        self.Wxh = np.random.randn(hidden_layer, input_size) / 1000
        self.Why = np.random.randn(output_size, hidden_layer) / 1000

        self.Bh = np.zeros((hidden_layer, 1))
        self.By = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))

        self.input = inputs
        self.h = {0: h}

        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.Bh)

            self.h[i + 1] = h

        y = self.Why @ h + self.By

        return y, h

    def backprop(self, d_y, rate = 2e-2):
        dL_dWhy = d_y @ self.h[len(self.input)].T
        dL_dBy = d_y

        dL_dWhh = np.zeros(self.Whh.shape)
        dL_dWxh = np.zeros(self.Wxh.shape)
        dL_dBh = np.zeros(self.Bh.shape)

        d_h = self.Why.T @ d_y

        for t in reversed(range(len(self.input))):
            temp = ((1 - self.h[t + 1] ** 2) * d_h)

            dL_dBh += temp

            dL_dWhh += temp @ self.h[t].T

            dL_dWxh += temp @ self.input[t].T

            dL_dh = self.Whh @ temp

        for d in [dL_dWxh, dL_dWhh, dL_dWhy, dL_dBh, dL_dBy]:
            np.clip(d, -1, 1, out = d)

        self.Whh -= rate * dL_dWhh
        self.Wxh -= rate * dL_dWxh
        self.Why -= rate * dL_dWhy
        self.Bh -= rate * dL_dBh
        self.By -= rate * dL_dBy
