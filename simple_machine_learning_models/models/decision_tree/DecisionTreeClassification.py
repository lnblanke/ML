# A simple decision tree model for classification using CART algorithm
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: DecisionTreeClassification.py

from scipy.stats import mode
import numpy as np
from .tree_node import TreeNode
class DecisionTreeClassification:
    name = "Decision Tree for Classification"

    def __init__(self, n_feature, max_depth):
        self.n_feature = n_feature
        self.max_depth = max_depth
        self.tree = [TreeNode(None, None, None)] * (2 ** (self.max_depth + 1))

    def _build_tree(self, layer, node, data, label, weight):
        if np.sum(label) == len(label) or np.sum(label) == 0 or layer == self.max_depth:
            self.tree[node] = TreeNode(None, None, mode(label)[0])
            return

        best_feature = 0
        best_thres = 0
        best_loss = 1e5

        for feature in range(self.n_feature):
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

        self.tree[node] = TreeNode(best_feature, best_thres, mode(label)[0])

        left_x = np.array([data[i] for i in range(len(data)) if (data[i][best_feature] < best_thres)])
        right_x = np.array([data[i] for i in range(len(data)) if (data[i][best_feature] >= best_thres)])
        left_y = np.array([label[i] for i in range(len(data)) if (data[i][best_feature] < best_thres)])
        right_y = np.array([label[i] for i in range(len(data)) if (data[i][best_feature] >= best_thres)])
        left_w = np.array([weight[i] for i in range(len(data)) if (data[i][best_feature] < best_thres)])
        right_w = np.array([weight[i] for i in range(len(data)) if (data[i][best_feature] >= best_thres)])

        if len(left_y) == 0 or len(right_y) == 0:
            self.tree[node] = TreeNode(None, None, mode(label)[0])
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
        if self.tree[node].feature == None:
            return np.full(len(data), self.tree[node].classes)

        feature = self.tree[node].feature
        thres = self.tree[node].thres
        pred = np.zeros(len(data))

        left_x = [data[i] for i in range(len(data)) if (data[i][feature]) < thres]
        left_idx = [i for i in range(len(data)) if (data[i][feature]) < thres]
        right_x = [data[i] for i in range(len(data)) if (data[i][feature]) >= thres]
        right_idx = [i for i in range(len(data)) if (data[i][feature]) >= thres]

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
