# A simple softmax regression model
# @Time: 3/9/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: SoftmaxRegression.py

import numpy as np
from ..Classifier import Classifier


class SoftmaxRegression(Classifier):
    name = "softmax regression"

    def __init__(self, n_features, n_classes, learning_rate = .1):
        super().__init__(n_features)
        self.n_classes = n_classes
        self.rate = learning_rate
        self.weight = np.random.random((self.n_classes, self.n_features))

    def train(self, train_x, train_y):
        count = 0
        prev_loss = 0

        while True:
            gradient = np.zeros((self.n_classes, self.n_features))
            loss = 0

            for i in range(len(train_y)):
                pred = np.zeros(self.n_classes)

                for j in range(self.n_classes):
                    pred[j] = np.exp(np.dot(self.weight[j], train_x[i]))

                for j in range(self.n_classes):
                    loss += -1 * (train_y[i] == j) * np.log(pred[j] / np.sum(pred))
                    gradient[j] += train_x[i] * ((train_y[i] == j) - pred[j] / np.sum(pred))

            self.weight -= -1 / len(train_y) * self.rate * gradient

            count += 1

            print("Epoch: %d Loss: %.5f" % (count, loss))

            if np.abs(loss - prev_loss) < 1:
                break

            prev_loss = loss

    def predict(self, test_x):
        pred = []

        for i in range(len(test_x)):
            pred.append(np.argmax(np.dot(self.weight, test_x[i])))

        return np.asarray(pred)
