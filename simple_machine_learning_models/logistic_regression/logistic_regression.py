# A simple logistic regression model for classification
# @Time: 3/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: logistic_regression.py

import numpy as np

def sigmoid(num):
    return 1.0 / (1 + np.exp(-1 * num))

class LR:
    def __init__(self, features, learning_rate = .00005):
        self.features = features
        self.rate = learning_rate
        self.weight = np.random.rand(self.features)

    def train(self, train_data, train_label):
        count = 0

        while True:
            gradient = 0

            for i in range(len(train_label)):
                gradient += (train_label[i] - sigmoid(np.dot(train_data[i], self.weight))) * train_data[i]

            self.weight += self.rate * gradient
            result = sigmoid(np.dot(train_data, self.weight))
            correct = 0

            print(gradient)

            for i in range(len(train_label)):
                if (result[i] >= 0.5) == train_label[i]:
                    correct += 1

            count += 1

            print("Epoch: %d Accuracy: %.5f" % (count, correct / len(train_label)))

            if correct / len(train_label) > .95:
                break

    def predict(self, test_data):
        pred = sigmoid(np.dot(test_data, self.weight)) >= .5

        return pred
