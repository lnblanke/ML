# A stochastic gradient descend version of linear regression
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: BGD.py

import numpy as np

class SGD:
    def __init__(self, features, learning_rate = .1):
        self.features = features
        self.weight = np.random.rand(self.features)
        self.rate = learning_rate

    def train(self, train_data, train_label):
        count = 0

        while True:
            index = int(np.random.rand() * len(train_label))
            gradient = (np.dot(train_data[index], self.weight) - train_label[index]) * train_data[index]

            self.weight -= self.rate * gradient

            cost = 0.5 * np.sum((np.dot(train_data, self.weight) - train_label) ** 2)

            count += 1

            print("Epoch: %d Loss: %.5f" % (count, cost))

            if count >= 500:
                break

    def predict(self, test_data):
        pred = np.dot(test_data, self.weight)

        return pred
