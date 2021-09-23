# A locally weight regression version of linear regression
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: BGD.py

import numpy as np
from . import normal_equation

class LWR:
    def __init__(self, features):
        self.features = features
        self.weight = np.random.rand(self.features)

    def train(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label

    def predict(self, test_data):
        pred = []
        for i in range(len(test_data)):
            bias = np.empty((len(self.train_data), 1))

            for j in range(len(self.train_data)):
                bias[j] = np.exp(-1 * np.linalg.norm(self.train_data[j] - test_data[i]) ** 2)

            ne = NE(self.features)
            ne.train(np.multiply(bias, self.train_data), self.train_label)

            pred.append(ne.predict(test_data[i]))
        return pred
