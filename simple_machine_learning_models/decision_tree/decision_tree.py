# A simple decision tree model for classification using CART algorithm
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: decision_tree.py

from scipy.stats import mode
import numpy as np
from tree_node import TreeNode

class DT:
    def __init__(self, features, max_depth):
        self.features = features
        self.max_depth = max_depth
        self.tree = [TreeNode(None, None, None)] * (2 ** (self.max_depth + 1))

    def _build_tree(self, layer, node, data, label):
        if layer == self.max_depth:
            self.tree[node] = TreeNode(None, None, mode(label)[0])
            return

        best_feature = 0
        best_thres = 0
        best_loss = 1e5

        for feature in range(self.features):
            num_left = [0, 0]
            num_right = [np.sum(label == cls) for cls in range(2)]

            thres, cls = zip(*sorted(zip(data[:, feature], label)))

            for i in range(len(label) - 1):
                num_left[cls[i]] += 1
                num_right[cls[i]] -= 1

                gini_left = 1 - np.sum((num / (i + 1)) ** 2 for num in num_left)
                gini_right = 1 - np.sum((num / (len(label) - i - 1)) ** 2 for num in num_right)

                loss = (i + 1) / len(label) * gini_left + (len(label) - i - 1) / len(label) * gini_right

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_thres = (thres[i + 1] - thres[i]) / 2

        self.tree[node] = TreeNode(best_feature, best_thres, mode(label)[0])

        left_data = np.array([data[i] for i in range(len(data)) if (data[i][best_feature] < best_thres)])
        right_data = np.array([data[i] for i in range(len(data)) if (data[i][best_feature] >= best_thres)])
        left_label = np.array([label[i] for i in range(len(data)) if (data[i][best_feature] < best_thres)])
        right_label = np.array([label[i] for i in range(len(data)) if (data[i][best_feature] >= best_thres)])

        if len(left_data) == 0 or len(right_data) == 0:
            self.tree[node] = TreeNode(None, None, mode(label)[0])
            return

        self._build_tree(layer + 1, node * 2, left_data, left_label)
        self._build_tree(layer + 1, node * 2 + 1, right_data, right_label)

    def train(self, train_x, train_y):
        self._build_tree(1, 1, train_x, train_y)

    def _classify(self, data, node):
        if self.tree[node].feature == None:
            return np.full(len(data), self.tree[node].classes)

        feature = self.tree[node].feature
        thres = self.tree[node].thres
        pred = np.zeros(len(data))

        left_data = [data[i] for i in range(len(data)) if (data[i][feature]) < thres]
        left_idx = [i for i in range(len(data)) if (data[i][feature]) < thres]
        right_data = [data[i] for i in range(len(data)) if (data[i][feature]) >= thres]
        right_idx = [i for i in range(len(data)) if (data[i][feature]) >= thres]

        left_pred = self._classify(left_data, node * 2)
        right_pred = self._classify(right_data, node * 2 + 1)

        for i in range(len(left_idx)):
            pred[left_idx[i]] = left_pred[i]

        for i in range(len(right_idx)):
            pred[right_idx[i]] = right_pred[i]

        return pred

    def predict(self, test_x):
        pred = self._classify(test_x, 1)

        return pred