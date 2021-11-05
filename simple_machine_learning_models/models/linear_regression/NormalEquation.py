# A batch gradient descend version of linear regression
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: NormalEquation.py

import numpy as np
from ..Regressor import Regressor


class NormalEquation(Regressor):
    name = "normal equation"

    def __init__(self, n_features):
        super().__init__(n_features)
        self.weight = np.random.rand(self.n_features)

    def train(self, train_x, train_y, verbose = 0):
        self.weight = np.dot(np.dot(np.linalg.inv(np.dot(train_x.transpose(), train_x)), train_x.transpose()),
                             train_y)

    def predict(self, test_x):
        pred = np.dot(test_x, self.weight)

        return pred
