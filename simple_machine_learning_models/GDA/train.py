# Train a GDA classifier
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: train.py

from GDA import GDA
from tools import *
import time
import numpy as np

if __name__ == '__main__':
    init = time.time()

    train_data, test_data, train_label, test_label = get_classification_data(sep = 4)

    gda = GDA(len(train_data[0]))
    gda.train(train_data, train_label)
    pred = gda.predict(test_data)

    print("Correction: %.2f" % (np.sum(pred == test_label) / len(test_label) * 100) + "%")
    print("Time: %.5fms" % ((time.time() - init) * 1000))

    show(test_data, pred, test_label, "GDA")
