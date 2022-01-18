# A stochastic gradient descend version of linear regression
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: StochasticGradientDescend.py

import numpy as np
from ..Regressor import Regressor


class StochasticGradientDescend(Regressor):
    name = "stochastic gradient descend"

    def __init__(self, n_features, learning_rate = .1):
        super().__init__(n_features)
        self.weight = np.random.rand(self.n_features)
        self.rate = learning_rate

    def train(self, train_x, train_y, verbose = 1):
        count = 0
        prev_loss = 0

        while True:
            index = int(np.random.rand() * len(train_y))
            gradient = (np.dot(train_x[index], self.weight) - train_y[index]) * train_x[index]

            self.weight -= self.rate * gradient
            loss = 0.5 * np.sum((np.dot(train_x, self.weight) - train_y) ** 2) / len(train_y)

            count += 1

            if verbose != 0 and count % 100 == 0:
                print("Epoch: %d Loss: %.5f" % (count, loss))

            if np.abs(loss - prev_loss) < 1:
                break

            prev_loss = loss

    def predict(self, test_x):
        pred = np.dot(test_x, self.weight)

        return pred
