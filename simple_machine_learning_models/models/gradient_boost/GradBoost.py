# A simple gradient boosting regression tree model
# @Time: 10/28/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: GradBoost.py

import numpy as np
from ..decision_tree import DecisionTreeRegression
from tools import show_trendline
import matplotlib.pyplot as plt

class GradBoost:
    name = "Gradient Boost"

    def __init__(self, n_feature, n_classifier = 10, learning_rate = .01):
        self.n_feature = n_feature
        self.n_classifier = n_classifier
        self.classifiers = []

        for i in range(n_classifier):
            self.classifiers.append(DecisionTreeRegression(self.n_feature, 3))

        self.learning_rate = learning_rate

    def train(self, train_x, train_y):
        y = np.copy(train_y)
        for i in range(self.n_classifier):
            self.classifiers[i].train(train_x, y)

            pred = self.classifiers[i].predict(train_x)

            print("Epoch: {:d} MSE: {:f}".format(i + 1, np.sum((y - pred) ** 2) / len(y)))

            y -= pred

    def predict(self, test_x):
        pred = np.sum(classifier.predict(test_x) for classifier in self.classifiers)

        return pred
