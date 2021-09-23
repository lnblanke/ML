# Some simple linear regression models
# @Time: 3/1/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: linear_regression.py

import numpy as np
import time
import matplotlib.pyplot as plt
from tools import *
from BGD import BGD
from SGD import SGD
from normal_equation import NE
from locally_weight_regression import LWR

def show(test_data, test_label, model, model_name):
    plt.scatter(np.reshape(np.arange(-5, 5, .01), (-1, 1)),
        model.predict(np.reshape(np.arange(-5, 5, .01), (-1, 1))), c = "red", s = 1)

    plt.scatter(test_data[:, 0], test_label, c = "black")
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.title(model_name)

    plt.show()

if __name__ == '__main__':
    train_data, test_data, train_label, test_label = get_regression_data()

    # BGD
    init = time.time()

    bgd = BGD(len(train_data[0]))

    print("Batch Gadient Descent:")

    bgd.train(train_data, train_label)

    pred = bgd.predict(test_data)

    mse = np.sum((pred - test_label) ** 2)

    print("Loss:", mse / len(test_label))
    print("Time: %.5fms" % ((time.time() - init) * 1000))

    show(test_data, test_label, bgd, "BGD")

    # SGD
    init = time.time()

    print("Stochastic Gradient Descent:")

    sgd = SGD(len(train_data[0]))
    sgd.train(train_data, train_label)
    pred = sgd.predict(test_data)

    mse = np.sum((pred - test_label) ** 2)

    print("Loss:", mse / len(test_label))
    print("Time: %.5fms" % ((time.time() - init) * 1000))

    show(test_data, test_label, sgd, "SGD")

    # Normal Equation
    init = time.time()

    ne = NE(len(train_data[0]))
    ne.train(train_data, train_label)
    pred = ne.predict(test_data)

    mse = np.sum((pred - test_label) ** 2)

    print("Normal Equation:")
    print("Loss:", mse / len(test_label))
    print("Time: %.5fms" % ((time.time() - init) * 1000))

    show(test_data, test_label, ne, "Normal Equation")

    # Locally Weighted Regression
    init = time.time()

    lwr = LWR(len(train_data[0]))
    lwr.train(train_data, train_label)
    pred = lwr.predict(test_data)

    print("Locally Weighted Regression:")

    mse = np.sum((pred - test_label) ** 2)

    print("Loss:", mse / len(test_label))
    print("Time: %.5fms" % ((time.time() - init) * 1000))

    show(train_data, train_label, lwr, "Locally Weighted Regression")
