# A locally weight regression version of linear regression
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: LocallyWeightedRegression.py

import numpy as np
from ..Regressor import Regressor


class LocallyWeightedRegression(Regressor):
    name = "locally weighted regression"

    def __init__(self, n_features, learning_rate):
        super().__init__(n_features)
        self.rate = learning_rate

    def train(self, train_x, train_y, verbose = 0):
        self.train_x = train_x
        self.train_y = train_y

    def predict(self, test_x):
        pred = []
        for i in range(len(test_x)):
            weight = np.empty((len(self.train_x), 1))

            for j in range(len(self.train_x)):
                weight[j] = np.exp(-1 * np.linalg.norm(self.train_x[j] - test_x[i]) ** 2)

            count = 0
            prev_loss = 0

            w = np.random.rand(self.n_features)

            while True:
                gradient = np.dot(np.multiply(weight.flatten(), (np.dot(self.train_x, w) - self.train_y)), self.train_x)

                w -= self.rate * gradient / len(self.train_y)

                loss = 0.5 * np.dot(np.transpose(weight), (np.dot(self.train_x, w) - self.train_y) ** 2)

                count += 1

                if np.abs(loss - prev_loss) < 1:
                    break

                prev_loss = loss

            pred.append(np.dot(test_x[i], w))

        return pred
