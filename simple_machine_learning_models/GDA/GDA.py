# A simple Gaussian Discriminant Analysis model for classification
# @Time: 3/12/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: GDA.py

import numpy as np

class GDA:
    def __init__(self, features):
        self.features = features
        self.phi = 0.0
        self.mu0 = np.zeros(self.features)
        self.mu1 = np.zeros(self.features)
        self.sigma = np.zeros((self.features, self.features))

    def train(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label

        num_pos = 0.0
        num_neg = 0.0

        for i in range(len(train_label)):
            self.phi += (train_label[i] == 1)

            num_pos += (train_label[i] == 1)
            num_neg += (train_label[i] == 0)

            self.mu0 += (train_label[i] == 0) * train_data[i]
            self.mu1 += (train_label[i] == 1) * train_data[i]

        self.phi /= len(train_label)
        self.mu0 /= num_neg
        self.mu1 /= num_pos

        for i in range(len(train_label)):
            if train_label[i] == 0:
                self.sigma += np.dot(np.reshape(train_data[i] - self.mu0, (len(train_data[i]), 1)),
                    np.reshape(train_data[i] - self.mu0, (1, len(train_data[i]))))
            else:
                self.sigma += np.dot(np.reshape(train_data[i] - self.mu1, (len(train_data[i]), 1)),
                    np.reshape(train_data[i] - self.mu1, (1, len(train_data[i]))))

    def predict(self, test_data):
        pred = []

        for i in range(len(test_data)):
            p_x_y_0 = 1 / ((2 * np.pi) ** len(self.train_data[0]) * np.sqrt(
                np.linalg.det(self.sigma))) * np.exp(
                -.5 * np.dot(np.dot(np.transpose(test_data[i] - self.mu0), np.linalg.inv(self.sigma)),
                    (test_data[i] - self.mu0)))
            p_x_y_1 = 1 / ((2 * np.pi) ** len(self.train_data[0]) * np.sqrt(
                np.linalg.det(self.sigma))) * np.exp(
                -.5 * np.dot(np.dot(np.transpose(test_data[i] - self.mu1), np.linalg.inv(self.sigma)),
                    (test_data[i] - self.mu1)))

            p_0 = p_x_y_0 * (1 - self.phi) / (p_x_y_0 + p_x_y_1)
            p_1 = p_x_y_1 * self.phi / (p_x_y_0 + p_x_y_1)

            pred.append(p_0 <= p_1)

        return pred
