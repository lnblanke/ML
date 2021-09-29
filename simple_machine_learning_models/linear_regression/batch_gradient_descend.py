# A batch gradient descend version of linear regression
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: batch_gradient_descend.py

import numpy as np

class BatchGradientDescend:
    def __init__(self, n_feature, learning_rate = .1):
        self.n_feature = n_feature
        self.weight = np.random.rand(self.n_feature)
        self.rate = learning_rate

    def train(self, train_x, train_y):
        count = 0

        while True:
            gradient = np.zeros(self.n_feature)

            for i in range(len(train_x)):
                gradient += (np.dot(train_x[i], self.weight) - train_y[i]) * train_x[i]

            self.weight -= self.rate / len(train_y) * gradient

            cost = 0.5 * np.sum((np.dot(train_x, self.weight) - train_y) ** 2)

            count += 1

            print("Epoch: %d Loss: %.5f" % (count, cost))

            if count >= 200:
                break

    def predict(self, test_x):
        pred = np.dot(test_x, self.weight)

        return pred
