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

        if max_depth > 50:
            raise ValueError("The max depth is too large!")

        self.max_depth = max_depth
        self.root = TreeNode(None, None, None, None, None)

    def _split(self, array: np.ndarray, idx_left: np.ndarray, idx_right) -> (np.ndarray, np.ndarray):
        left = np.array([array[i] for i in idx_left])
        right = np.array([array[i] for i in idx_right])

        return left, right

    def _build_tree(self, layer: int, node: TreeNode, data: np.ndarray, label: np.ndarray, weight: np.ndarray):
        if layer == self.max_depth:
            node.value = np.mean(label)
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

        left_idx = np.array([i for i in range(len(data)) if (data[i][best_feature] < best_thres)])
        right_idx = np.array([i for i in range(len(data)) if (data[i][best_feature] >= best_thres)])

        left_x, right_x = self._split(data, left_idx, right_idx)
        left_y, right_y = self._split(label, left_idx, right_idx)
        left_w, right_w = self._split(weight, left_idx, right_idx)

        if len(left_y) == 0 or len(right_y) == 0:
            node.value = np.mean(label)
            return

        node.threshold = best_thres
        node.feature = best_feature
        node.value = np.mean(label)
        node.left_child = TreeNode(None, None, None, None, None)
        node.right_child = TreeNode(None, None, None, None, None)

        self._build_tree(layer + 1, node.left_child, left_x, left_y, left_w)
        self._build_tree(layer + 1, node.right_child, right_x, right_y, right_w)

    def train(self, train_x, train_y, weight = None):
        if weight is not None:
            weight = weight
        else:
            weight = np.ones(len(train_y))

        self._build_tree(0, self.root, train_x, train_y, weight)

    def _regression(self, data: np.ndarray, node: TreeNode):
        if node.feature is None:
            return np.full(len(data), node.value)

        feature = node.feature
        thres = node.threshold
        pred = np.zeros(len(data))

        left_idx = np.array([i for i in range(len(data)) if (data[i][feature] < thres)])
        right_idx = np.array([i for i in range(len(data)) if (data[i][feature] >= thres)])

        left_x, right_x = self._split(data, left_idx, right_idx)

        left_pred = self._regression(left_x, node.left_child)
        right_pred = self._regression(right_x, node.right_child)

        for i in range(len(left_idx)):
            pred[left_idx[i]] = left_pred[i]

        for i in range(len(right_idx)):
            pred[right_idx[i]] = right_pred[i]

        return pred

    def predict(self, test_x):
        pred = self._regression(test_x, self.root)

        return pred
