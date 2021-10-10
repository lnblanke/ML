# A simple logistic regression model for classification
# @Time: 3/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: LogisticRegression.py

import numpy as np

def sigmoid(num):
    return 1.0 / (1 + np.exp(-1 * num))

class LogisticRegression:
    name = "Logistic Regression"

    def __init__(self, n_feature, learning_rate = .00005):
        self.n_feature = n_feature
        self.rate = learning_rate
        self.weight = np.random.rand(self.n_feature)

    def train(self, train_x, train_y):
        count = 0
        prev_accuracy = 0

        while True:
            gradient = 0

            for i in range(len(train_y)):
                gradient += (train_y[i] - sigmoid(np.dot(train_x[i], self.weight))) * train_x[i]

            self.weight += self.rate * gradient
            result = sigmoid(np.dot(train_x, self.weight))
            correct = 0

            for i in range(len(train_y)):
                if (result[i] >= 0.5) == train_y[i]:
                    correct += 1

            count += 1

            accuracy = correct / len(train_y)

            print("Epoch: {:d} Accuracy: {:.2%}".format(count, accuracy))

            if accuracy == prev_accuracy:
                break

            prev_accuracy = accuracy

    def predict(self, test_x):
        pred = sigmoid(np.dot(test_x, self.weight)) >= .5

        return pred
