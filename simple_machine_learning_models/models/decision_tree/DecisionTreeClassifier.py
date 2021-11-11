# A simple decision tree model for classification using CART algorithm
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: DecisionTreeClassifier.py

from scipy.stats import mode
import numpy as np
from .tree_node import TreeNode
from ..Classifier import Classifier


class DecisionTreeClassifier(Classifier):
    name = "decision tree classifier"

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
        if np.sum(label) == len(label) or np.sum(label) == 0 or layer == self.max_depth:
            node.value = mode(label)[0]
            return

        best_feature = 0
        best_thres = 0
        best_loss = 1e5

        for feature in range(self.n_features):
            num_left = [0, 0]

            num_right = [0, 0]
            for i in range(len(weight)):
                num_right[label[i]] += weight[i]

            thres, cls, w = zip(*sorted(zip(data[:, feature], label, weight)))

            for i in range(len(label) - 1):
                num_left[cls[i]] += w[i]
                num_right[cls[i]] -= w[i]

                gini_left = 1 - np.sum((num / np.sum(num_left)) ** 2 for num in num_left)
                gini_right = 1 - np.sum((num / np.sum(num_right)) ** 2 for num in num_right)

                loss = (i + 1) / len(label) * gini_left + (len(label) - i - 1) / len(label) * gini_right

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
            node.value = mode(label)[0]
            return

        node.threshold = best_thres
        node.feature = best_feature
        node.value = mode(label)[0]
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

    def _classify(self, data: np.ndarray, node: TreeNode):
        if node.feature is None:
            return np.full(len(data), node.value)

        feature = node.feature
        thres = node.threshold
        pred = np.zeros(len(data))

        left_idx = np.array([i for i in range(len(data)) if (data[i][feature] < thres)])
        right_idx = np.array([i for i in range(len(data)) if (data[i][feature] >= thres)])

        left_x, right_x = self._split(data, left_idx, right_idx)

        left_pred = self._classify(left_x, node.left_child)
        right_pred = self._classify(right_x, node.right_child)

        for i in range(len(left_idx)):
            pred[left_idx[i]] = left_pred[i]

        for i in range(len(right_idx)):
            pred[right_idx[i]] = right_pred[i]

        return pred

    def predict(self, test_x):
        pred = self._classify(test_x, self.root)

        return pred
