# A simple softmax regression model
# @Time: 3/9/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: SoftmaxRegression.py

import numpy as np

class SoftmaxRegression:
    name = "Softmax Regression"

    def __init__(self, n_feature, n_class, learning_rate = .1):
        self.n_feature = n_feature
        self.n_class = n_class
        self.rate = learning_rate
        self.weight = np.random.random((self.n_class, self.n_feature))

    def train(self, train_x, train_y):
        count = 0
        prev_loss = 0

        while True:
            gradient = np.zeros((self.n_class, self.n_feature))
            loss = 0

            for i in range(len(train_y)):
                pred = np.zeros(self.n_class)

                for j in range(self.n_class):
                    pred[j] = np.exp(np.dot(self.weight[j], train_x[i]))

                for j in range(self.n_class):
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

        return pred
