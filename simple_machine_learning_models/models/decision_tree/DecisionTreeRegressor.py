# A simple decision tree model for regression
# @Time: 10/30/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: DecisionTreeRegressor.py

import numpy as np
from .tree_node import TreeNode
from ..Regressor import Regressor


class DecisionTreeRegressor(Regressor):
    name = "decision tree regressor"

    def __init__(self, n_features, max_depth: int):
        super().__init__(n_features)
        self.max_depth = max_depth
        self.tree = [TreeNode(None, None, None)] * (2 ** (self.max_depth + 1))

    def _build_tree(self, layer, node, data, label, weight):
        if layer == self.max_depth:
            self.tree[node] = TreeNode(None, None, np.mean(label))
            return

        best_feature = 0
        best_thres = 0
        best_loss = 1e5

        for feature in range(self.n_features):
            thres = np.array(sorted(data[:, feature]))

            for i in range(len(label) - 1):
                y_left = np.array([label[j] for j in range(len(label)) if data[j][feature] <= thres[i]])
                y_right = np.array([label[j] for j in range(len(label)) if data[j][feature] > thres[i]])
                avg_left = np.mean(y_left)
                avg_right = np.mean(y_right)

                loss = np.sum((y_left - avg_left) ** 2) / len(label) + np.sum((y_right - avg_right) ** 2) / len(label)

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_thres = (thres[i + 1] + thres[i]) / 2

        self.tree[node] = TreeNode(best_feature, best_thres, np.mean(label))

        left_x = np.array([data[i] for i in range(len(data)) if (data[i][best_feature] <= best_thres)])
        right_x = np.array([data[i] for i in range(len(data)) if (data[i][best_feature] > best_thres)])
        left_y = np.array([label[i] for i in range(len(data)) if (data[i][best_feature] <= best_thres)])
        right_y = np.array([label[i] for i in range(len(data)) if (data[i][best_feature] > best_thres)])
        left_w = np.array([weight[i] for i in range(len(data)) if (data[i][best_feature] <= best_thres)])
        right_w = np.array([weight[i] for i in range(len(data)) if (data[i][best_feature] > best_thres)])

        if len(left_y) == 0 or len(right_y) == 0:
            self.tree[node] = TreeNode(None, None, np.mean(label))
            return

        self._build_tree(layer + 1, node * 2, left_x, left_y, left_w)
        self._build_tree(layer + 1, node * 2 + 1, right_x, right_y, right_w)

    def train(self, train_x, train_y, weight = None):
        if weight is not None:
            weight = weight
        else:
            weight = np.ones(len(train_y))

        self._build_tree(0, 1, train_x, train_y, weight)

    def _classify(self, data, node):
        if self.tree[node].feature is None:
            return np.full(len(data), self.tree[node].classes)

        feature = self.tree[node].feature
        thres = self.tree[node].thres
        pred = np.zeros(len(data))

        left_x = [data[i] for i in range(len(data)) if (data[i][feature]) <= thres]
        left_idx = [i for i in range(len(data)) if (data[i][feature]) <= thres]
        right_x = [data[i] for i in range(len(data)) if (data[i][feature]) > thres]
        right_idx = [i for i in range(len(data)) if (data[i][feature]) > thres]

        left_pred = self._classify(left_x, node * 2)
        right_pred = self._classify(right_x, node * 2 + 1)

        for i in range(len(left_idx)):
            pred[left_idx[i]] = left_pred[i]

        for i in range(len(right_idx)):
            pred[right_idx[i]] = right_pred[i]

        return pred

    def predict(self, test_x):
        pred = self._classify(test_x, 1)

        return pred
