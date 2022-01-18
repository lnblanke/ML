# A simple naive Bayes model with Laplace smoothing for classification
# @Time: 3/12/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: NaiveBayes.py

import numpy as np
from ..Classifier import Classifier


class NaiveBayes(Classifier):
    name = "naive Bayes"

    def __init__(self, n_features):
        super().__init__(n_features)
        self.phi_j_y_1 = np.ones((self.n_features, 20))
        self.phi_j_y_0 = np.ones((self.n_features, 20))

    def train(self, train_x, train_y):
        pos = 10
        neg = 10

        for i in range(len(train_y)):
            if train_y[i] == 0:
                neg += 1
                self.phi_j_y_0[0][int(train_x[i][0]) + 5] += 1
                self.phi_j_y_0[1][int(train_x[i][1]) + 5] += 1
            else:
                pos += 1
                self.phi_j_y_1[0][int(train_x[i][0]) + 5] += 1
                self.phi_j_y_1[1][int(train_x[i][1]) + 5] += 1

        self.phi_j_y_0 /= neg
        self.phi_j_y_1 /= pos
        self.phi_y_0 = (neg - 9) / (neg + pos - 18)
        self.phi_y_1 = (pos - 9) / (pos + neg - 18)

    def predict(self, test_x):
        pred = []

        for i in range(len(test_x)):
            phi_x_y_1 = self.phi_j_y_1[0][int(test_x[i][0]) + 5] * self.phi_j_y_1[1][
                int(test_x[i][1]) + 5] * self.phi_y_1
            phi_x_y_0 = self.phi_j_y_0[0][int(test_x[i][0]) + 5] * self.phi_j_y_0[1][
                int(test_x[i][1]) + 5] * self.phi_y_0

            p_0 = phi_x_y_0 / (phi_x_y_0 + phi_x_y_1)
            p_1 = phi_x_y_1 / (phi_x_y_0 + phi_x_y_1)

            pred.append(p_1 >= p_0)

        return np.asarray(pred)
