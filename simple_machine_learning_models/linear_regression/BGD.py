# A batch gradient descend version of linear regression
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: BGD.py

import numpy as np

class BGD:
    def __init__(self, features, learning_rate = .1):
        self.features = features
        self.weight = np.random.rand(self.features)
        self.rate = learning_rate

    def train(self, train_data, train_label):
        count = 0

        while True:
            gradient = np.zeros(self.features)

            for i in range(len(train_data)):
                gradient += (np.dot(train_data[i], self.weight) - train_label[i]) * train_data[i]

            self.weight -= self.rate / len(train_label) * gradient

            cost = 0.5 * np.sum((np.dot(train_data, self.weight) - train_label) ** 2)

            count += 1

            print("Epoch: %d Loss: %.5f" % (count, cost))

            if count >= 200:
                break

    def predict(self, test_data):
        pred = np.dot(test_data, self.weight)

        return pred
