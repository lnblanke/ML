# A simple support vector machine for classification model
# @Time: 3/21/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: SVM.py

import numpy as np
from ..Classifier import Classifier
from .kernel import linear, poly, rbf, laplace


class SVM(Classifier):
    name = "SVM"

    def __init__(self, n_features, c = 1, kernel = "rbf"):
        super().__init__(n_features)
        
        self.param = self.x = self.y = None
        self.b = 0
        self.c = c

        if kernel is None or kernel == "linear":
            self.kernel = linear
        elif kernel == "poly":
            self.kernel = poly
        elif kernel == "rbf":
            self.kernel = rbf
        elif kernel == "laplace":
            self.kernel = laplace
        else:
            raise NotImplementedError()

    def train(self, train_x: np.ndarray, train_y: np.ndarray, epochs = 20):
        train_y = np.where(train_y, 1, -1)
        self.param = np.ones(len(train_y))
        self.x = train_x
        self.y = train_y

        for _ in range(epochs):
            for i in range(len(train_y) - 1):
                j = np.random.randint(i + 1, len(train_y))
                eps = self.param[i] * train_y[i] + self.param[j] * train_y[j]
                l1, l2 = self.param[i], self.param[j]

                self.param[j] += train_y[j] / (
                        self.kernel(train_x[i], train_x[i]) - 2 * self.kernel(train_x[i], train_x[j]) + self.kernel(
                    train_x[j], train_x[j])) * (
                                    np.sum(self.param * train_y * self.kernel(train_x, train_x[j])) - np.sum(
                                self.param * train_y * self.kernel(train_x, train_x[i])) + train_y[i] - train_y[j])

                if train_y[i] == train_y[j]:
                    self.param[j] = np.clip(self.param[j], max(0, l1 + l2), min(l1 + l2, self.c))
                else:
                    self.param[j] = np.clip(self.param[j], max(0, l2 - l1), min(self.c - l1 + l2, self.c))

                self.param[i] = (eps - train_y[j] * self.param[j]) * train_y[i]

        self.b = np.sum(train_y - np.sum(self.param * train_y)) / len(train_y)

    def predict(self, test_x: np.ndarray):
        pred = np.zeros(len(test_x))

        for i in range(len(test_x)):
            pred[i] = np.sum(self.param * self.y * self.kernel(self.x, test_x[i])) >= 0

        return pred
