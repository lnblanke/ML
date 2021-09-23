# Train a GDA classifier
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: train.py

from tools import *
import time
import numpy as np
from softmax_regression import SR

n_class = 4

if __name__ == '__main__':
    init = time.time()

    train_data, test_data, train_label, test_label = get_classification_data(sep = 3, classes = n_class)

    sr = SR(len(train_data[0]), classes = n_class)
    sr.train(train_data, train_label)
    pred = sr.predict(test_data)

    print("Correction: %.2f" % (np.sum(pred == test_label) / len(test_label) * 100) + "%")
    print("Time: %.5fms" % ((time.time() - init) * 1000))

    show(test_data, pred, test_label, "Naive Bayes")
