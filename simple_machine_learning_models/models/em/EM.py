# A simple expectation maximization unsupervised classification model
# @Time: 6/20/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: EM.py

import numpy as np
from ..Unsupervised import Unsupervised


class EM(Unsupervised):
    name = "EM"

    def __init__(self, n_features, n_classes = 2):
        super().__init__(n_features)
        self.n_classes = n_classes
        self.phi = np.random.rand(n_classes)
        self.mu = np.random.randn(n_classes, n_features)
        self.sigma = np.stack([np.identity(n_features)] * n_classes)

    def train(self, train_x, epochs = 50):
        w = np.zeros((len(train_x), self.n_classes))

        for _ in range(epochs):
            for j in range(self.n_classes):
                w[:, j] = np.diag(1 / ((2 * np.pi) ** (self.n_features / 2) * np.sqrt(
                    np.linalg.det(self.sigma[j]))) * np.exp(
                    -.5 * ((train_x - self.mu[j]) @ np.linalg.inv(self.sigma[j]) @ (train_x - self.mu[j]).T)))

            w = w * self.phi / np.sum(w * self.phi, axis = 1, keepdims = True)

            self.phi = np.sum(w, axis = 0) / len(train_x)
            self.mu = (w.T @ train_x) / np.sum(w, axis = 0).reshape(-1, 1)

            self.sigma = np.zeros((self.n_classes, self.n_features, self.n_features))

            for j in range(self.n_classes):
                self.sigma[j] = (w[:, j] * (train_x - self.mu[j]).T @ (train_x - self.mu[j]))

            self.sigma /= np.sum(w, axis = 0).reshape(-1, 1, 1)

        return np.argmax(w, axis = 1)
