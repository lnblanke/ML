# A simple Gaussian Discriminant Analysis model for classification
# @Time: 3/12/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: GDA.py

import numpy as np

class GDA:
    def __init__(self, n_feature):
        self.n_feature = n_feature
        self.phi = 0.0
        self.mu0 = np.zeros(self.n_feature)
        self.mu1 = np.zeros(self.n_feature)
        self.sigma = np.zeros((self.n_feature, self.n_feature))

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
                self.sigma += np.dot(np.reshape(train_x[i] - self.mu0, (len(train_x[i]), 1)),
                    np.reshape(train_x[i] - self.mu0, (1, len(train_x[i]))))
            else:
                self.sigma += np.dot(np.reshape(train_x[i] - self.mu1, (len(train_x[i]), 1)),
                    np.reshape(train_x[i] - self.mu1, (1, len(train_x[i]))))

    def predict(self, test_x):
        pred = []

        for i in range(len(test_x)):
            p_x_y_0 = 1 / ((2 * np.pi) ** len(self.train_x[0]) * np.sqrt(
                np.linalg.det(self.sigma))) * np.exp(
                -.5 * np.dot(np.dot(np.transpose(test_x[i] - self.mu0), np.linalg.inv(self.sigma)),
                    (test_x[i] - self.mu0)))
            p_x_y_1 = 1 / ((2 * np.pi) ** len(self.train_x[0]) * np.sqrt(
                np.linalg.det(self.sigma))) * np.exp(
                -.5 * np.dot(np.dot(np.transpose(test_x[i] - self.mu1), np.linalg.inv(self.sigma)),
                    (test_x[i] - self.mu1)))

            p_0 = p_x_y_0 * (1 - self.phi) / (p_x_y_0 + p_x_y_1)
            p_1 = p_x_y_1 * self.phi / (p_x_y_0 + p_x_y_1)

            pred.append(p_0 <= p_1)

        return pred
