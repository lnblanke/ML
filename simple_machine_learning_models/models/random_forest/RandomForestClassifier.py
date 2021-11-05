# A simple random forest classifier using decision tree classifier
# @Time: 9/23/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: RandomForestClassifier.py
import numpy as np
from ..decision_tree import DecisionTreeClassifier

class RandomForestClassifier:
    name = "Random Forest Classifier"

    def __init__(self, n_feature, n_trees, max_depth = 1):
        self.n_feature = n_feature
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

        for i in range(self.n_trees):
            self.trees.append(DecisionTreeClassifier(self.n_feature, self.max_depth))

    def _bootstrap(self, data, label):
        indice = np.arange(len(data))
        choice = np.random.choice(indice, replace = True, size = len(data))

        sample_x = np.array([data[i] for i in choice])
        sample_y = np.array([label[i] for i in choice])

        return sample_x, sample_y

    def train(self, train_x, train_y):
        for tree in self.trees:
            sample_x, sample_y = self._bootstrap(train_x, train_y)

            tree.train(sample_x, sample_y)

    def predict(self, test_x):
        sep_pred = []

        for tree in self.trees:
            sep_pred.append(tree.predict(test_x))

        pred = np.max(sep_pred, axis = 0)

        return pred