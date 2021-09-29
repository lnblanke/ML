# Train a logistic regression classifier
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: train.py

from tools import *
import time
import numpy as np
from logistic_regression import LogisticRegression

if __name__ == '__main__':
    init = time.time()

    train_data, test_data, train_label, test_label = get_classification_data(sep = 4)

    lr = LogisticRegression(len(train_data[0]))
    lr.train(train_data, train_label)
    pred = lr.predict(test_data)

    print("Correction: %.2f" % (np.sum(pred == test_label) / len(test_label) * 100) + "%")
    print("Time: %.5fms" % ((time.time() - init) * 1000))

    show(test_data, pred, test_label, "LR")
