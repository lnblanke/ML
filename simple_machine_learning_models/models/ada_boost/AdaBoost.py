# A simple AdaBoost classifier with decision tree classifiers
# @Time: 10/5/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: AdaBoost.py

import numpy as np
from ..decision_tree import DecisionTreeClassifier
from ..Classifier import Classifier
from ..Ensemble import Ensemble


class AdaBoost(Ensemble, Classifier):
    name = "AdaBoost"

    def __init__(self, n_features, n_predictor = 10, max_depth = 1, learning_rate = .1):
        super().__init__(n_features, n_predictor, DecisionTreeClassifier, n_features, max_depth)

        self.weight = np.zeros(n_predictor)
        self.learning_rate = learning_rate

    def train(self, train_x, train_y):
        weights = np.full(len(train_y), 1 / len(train_y))

        for i in range(self.n_predictor):
            self.predictors[i].train(train_x, train_y, weight = weights)

            pred = self.predictors[i].predict(train_x)

            error_rate = np.dot(weights, np.abs(pred - train_y)) / np.sum(weights)

            alpha = self.learning_rate * np.log((1 - error_rate) / error_rate)

            weights *= np.exp(alpha * (pred != train_y))

            self.weight[i] = alpha

            print("Epoch: {:d} Accuracy: {:.2%}".format(i + 1, 1 - np.sum(np.abs(pred - train_y)) / len(train_y)))

    def predict(self, test_x):
        votes = np.zeros((len(test_x), 2))

        for i in range(self.n_predictor):
            pred_i = self.predictors[i].predict(test_x)

            for j in range(len(test_x)):
                votes[j][int(pred_i[j])] += self.weight[i]

        pred = np.argmax(votes, axis = 1)

        return pred
