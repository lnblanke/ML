# A simple softmax regression model
# @Time: 3/9/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: softmax_regression.py

import numpy as np

class SR:
    def __init__(self, features, classes, learning_rate = .1):
        self.features = features
        self.classes = classes
        self.rate = learning_rate
        self.weight = np.random.random((self.classes, self.features))

    def train(self, train_data, train_label):
        count = 0

        while True:
            gradient = np.zeros((self.classes, self.features))
            cost = 0

            for i in range(len(train_label)):
                pred = np.zeros(self.classes)

                for j in range(self.classes):
                    pred[j] = np.exp(np.dot(self.weight[j], train_data[i]))

                for j in range(self.classes):
                    cost += -1 * (train_label[i] == j) * np.log(pred[j] / np.sum(pred))
                    gradient[j] += train_data[i] * ((train_label[i] == j) - pred[j] / np.sum(pred))

            self.weight -= -1 / len(train_label) * self.rate * gradient

            count += 1

            print("Epoch: %d Loss: %.5f" % (count, cost))

            if cost < 500:
                break

    def predict(self, test_data):
        pred = []

        for i in range(len(test_data)):
            pred.append(np.argmax(np.dot(self.weight, test_data[i])))

        return pred
