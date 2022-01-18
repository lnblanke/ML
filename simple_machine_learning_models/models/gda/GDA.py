# A simple Gaussian Discriminant Analysis model for classification
# @Time: 3/12/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: GDA.py

import numpy as np
from ..Classifier import Classifier


class GDA(Classifier):
    name = "GDA"

    def __init__(self, n_features):
        super().__init__(n_features)
        self.phi = 0.0
        self.mu0 = np.zeros(self.n_features)
        self.mu1 = np.zeros(self.n_features)
        self.sigma = np.zeros((self.n_features, self.n_features))

    def train(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

        num_pos = 0.0
        num_neg = 0.0

        for i in range(len(train_y)):
            self.phi += (train_y[i] == 1)

            num_pos += (train_y[i] == 1)
            num_neg += (train_y[i] == 0)

            self.mu0 += (train_y[i] == 0) * train_x[i]
            self.mu1 += (train_y[i] == 1) * train_x[i]

        self.phi /= len(train_y)
        self.mu0 /= num_neg
        self.mu1 /= num_pos

        for i in range(len(train_y)):
            if train_y[i] == 0:
                self.sigma += np.dot(np.transpose(np.asmatrix(train_x[i] - self.mu0)),
                                     np.asmatrix(train_x[i] - self.mu0))
            else:
                self.sigma += np.dot(np.transpose(np.asmatrix(train_x[i] - self.mu1)),
                                     np.asmatrix(train_x[i] - self.mu1))

        self.sigma /= len(train_y)

    def predict(self, test_x):
        pred = []

        for i in range(len(test_x)):
            p_x_y_0 = 1 / ((2 * np.pi) ** (self.n_features / 2) * np.sqrt(
                np.linalg.det(self.sigma))) * np.exp(
                -.5 * np.dot(np.dot(np.transpose(test_x[i] - self.mu0), np.linalg.inv(self.sigma)),
                             (test_x[i] - self.mu0)))
            p_x_y_1 = 1 / ((2 * np.pi) ** (self.n_features / 2) * np.sqrt(
                np.linalg.det(self.sigma))) * np.exp(
                -.5 * np.dot(np.dot(np.transpose(test_x[i] - self.mu1), np.linalg.inv(self.sigma)),
                             (test_x[i] - self.mu1)))

            p_0 = p_x_y_0 * (1 - self.phi) / (p_x_y_0 + p_x_y_1)
            p_1 = p_x_y_1 * self.phi / (p_x_y_0 + p_x_y_1)

            pred.append(p_0 <= p_1)

        return np.asarray(pred)
