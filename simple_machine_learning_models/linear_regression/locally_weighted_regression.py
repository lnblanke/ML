# A locally weight regression version of linear regression
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: locally_weighted_regression.py

import numpy as np
from normal_equation import NormalEquation

class LocallyWeightedRegression:
    def __init__(self, n_feature):
        self.n_feature = n_feature
        self.weight = np.random.rand(self.n_feature)

    def train(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def predict(self, test_x):
        pred = []
        for i in range(len(test_x)):
            bias = np.empty((len(self.train_x), 1))

            for j in range(len(self.train_x)):
                bias[j] = np.exp(-1 * np.linalg.norm(self.train_x[j] - test_x[i]) ** 2)

            ne = NormalEquation(self.n_feature)
            ne.train(np.multiply(bias, self.train_x), self.train_y)

            pred.append(ne.predict(test_x[i]))
        return pred
