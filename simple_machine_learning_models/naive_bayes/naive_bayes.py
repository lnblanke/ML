# A simple naive Bayes model with Laplace smoothing for classification
# @Time: 3/12/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: naive_bayes.py

import numpy as np

class NBC:
    def __init__(self, features):
        self.features = features
        self.phi_j_y_1 = np.ones((self.features, 20))
        self.phi_j_y_0 = np.ones((self.features, 20))

    def train(self, train_data, train_label):
        pos = 10
        neg = 10

        for i in range(len(train_label)):
            if train_label[i] == 0:
                neg += 1
                self.phi_j_y_0[0][int(train_data[i][0]) + 5] += 1
                self.phi_j_y_0[1][int(train_data[i][1]) + 5] += 1
            else:
                pos += 1
                self.phi_j_y_1[0][int(train_data[i][0]) + 5] += 1
                self.phi_j_y_1[1][int(train_data[i][1]) + 5] += 1

        self.phi_j_y_0 /= neg
        self.phi_j_y_1 /= pos
        self.phi_y_0 = (neg - 9) / (neg + pos - 18)
        self.phi_y_1 = (pos - 9) / (pos + neg - 18)

    def predict(self, test_data):
        pred = []

        for i in range(len(test_data)):
            phi_x_y_1 = self.phi_j_y_1[0][int(test_data[i][0]) + 5] * self.phi_j_y_1[1][
                int(test_data[i][1]) + 5] * self.phi_y_1
            phi_x_y_0 = self.phi_j_y_0[0][int(test_data[i][0]) + 5] * self.phi_j_y_0[1][
                int(test_data[i][1]) + 5] * self.phi_y_0

            p_0 = phi_x_y_0 / (phi_x_y_0 + phi_x_y_1)
            p_1 = phi_x_y_1 / (phi_x_y_0 + phi_x_y_1)

            pred.append(p_1 >= p_0)

        return pred
