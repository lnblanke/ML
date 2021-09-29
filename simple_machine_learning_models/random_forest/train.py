# Train a random forest classifier
# @Time: 9/25/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: train.py

from random_forest import RandomForest
from tools import *
import time
import numpy as np

if __name__ == '__main__':
    init = time.time()

    train_data, test_data, train_label, test_label = get_classification_data(sep = 1, clusters = 2)

    dt = RandomForest(len(train_data[0]), 10,  5)
    dt.train(train_data, train_label)
    pred = dt.predict(test_data)

    print("Correction: %.2f" % (np.sum(pred == test_label) / len(test_label) * 100) + "%")
    print("Time: %.5fms" % ((time.time() - init) * 1000))

    show(test_data, dt.predict(test_data), test_label, "Random Forest")
