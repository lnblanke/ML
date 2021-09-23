# A batch gradient descend version of linear regression
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: BGD.py

import numpy as np

class NE:
    def __init__(self, features):
        self.features = features
        self.weight = np.random.rand(self.features)

    def train(self, train_data, train_label):
        self.weight = np.dot(np.dot(np.linalg.inv(np.dot(train_data.transpose(), train_data)), train_data.transpose()),
            train_label)

    def predict(self, test_data):
        pred = np.dot(test_data, self.weight)

        return pred
