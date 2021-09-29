# A batch gradient descend version of linear regression
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: normal_equation.py

import numpy as np

class NormalEquation:
    def __init__(self, n_feature):
        self.n_feature = n_feature
        self.weight = np.random.rand(self.n_feature)

    def train(self, train_x, train_y):
        self.weight = np.dot(np.dot(np.linalg.inv(np.dot(train_x.transpose(), train_x)), train_x.transpose()),
            train_y)

    def predict(self, test_x):
        pred = np.dot(test_x, self.weight)

        return pred
