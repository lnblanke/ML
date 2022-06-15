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
        self.phi = np.zeros(2)
        self.mu = np.zeros((self.n_features, 2))
        self.sigma = np.zeros((self.n_features, self.n_features))

    def train(self, train_x, train_y):
        self.phi = np.array([len(train_y) - np.sum(train_y), np.sum(train_y)]) / len(train_y)

        for i in range(len(train_y)):
            self.mu[train_y[i]] += train_x[i]

        self.mu /= [len(train_y) - np.sum(train_y), np.sum(train_y)]

        for i in range(len(train_y)):
            self.sigma += np.dot(np.transpose(np.asmatrix(train_x[i] - self.mu[train_y[i]])),
                                 np.asmatrix(train_x[i] - train_y[i]))

        self.sigma /= len(train_y)

    def predict(self, test_x):
        p_x_y = np.zeros((len(test_x), 2))

        for i in range(2):
            p_x_y[:, i] = np.diag(1 / ((2 * np.pi) ** (self.n_features / 2) * np.sqrt(
                np.linalg.det(self.sigma))) * np.exp(
                -.5 * ((test_x - self.mu[i]) @ np.linalg.inv(self.sigma) @ (test_x - self.mu[i]).T)))

        p = p_x_y * self.phi / np.sum(p_x_y * self.phi, axis = 1, keepdims = True)

        return np.argmax(p, axis = 1)
