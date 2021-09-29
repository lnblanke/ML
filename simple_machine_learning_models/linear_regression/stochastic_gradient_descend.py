# A stochastic gradient descend version of linear regression
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: stochastic_gradient_descend.py

import numpy as np

class StochasticGradientDescend:
    def __init__(self, n_feature, learning_rate = .1):
        self.n_feature = n_feature
        self.weight = np.random.rand(self.n_feature)
        self.rate = learning_rate

    def train(self, train_x, train_y):
        count = 0

        while True:
            index = int(np.random.rand() * len(train_y))
            gradient = (np.dot(train_x[index], self.weight) - train_y[index]) * train_x[index]

            self.weight -= self.rate * gradient

            cost = 0.5 * np.sum((np.dot(train_x, self.weight) - train_y) ** 2)

            count += 1

            print("Epoch: %d Loss: %.5f" % (count, cost))

            if count >= 500:
                break

    def predict(self, test_x):
        pred = np.dot(test_x, self.weight)

        return pred
