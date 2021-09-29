# Train a decision tree classifier
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: train.py

from decision_tree import DecisionTree
from tools import *
import time
import numpy as np

if __name__ == '__main__':
    init = time.time()

    train_data, test_data, train_label, test_label = get_classification_data(sep = 1.5, clusters = 2)

    dt = DecisionTree(len(train_data[0]), 5)
    dt.train(train_data, train_label)
    pred = dt.predict(test_data)

    print("Correction: %.2f" % (np.sum(pred == test_label) / len(test_label) * 100) + "%")
    print("Time: %.5fms" % ((time.time() - init) * 1000))

    show(test_data, dt.predict(test_data), test_label, "Decision Tree")
