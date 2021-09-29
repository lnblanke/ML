# Train all the linear regression models
# @Time: 9/24/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: train.py.py

import time
import matplotlib.pyplot as plt
from tools import *
from linear_regression import LinearRegression

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

    type_list = ["Batch Gradient Descend", "Stochastic Gradient Descend", "Normal Equation", "Locally Weighted Regression"]

    for type in type_list:
        init = time.time()

        model = LinearRegression(type, len(train_data[0]))

        print(f"{type}:")

        model.train(train_data, train_label)

        pred = model.predict(test_data)

        mse = np.sum((pred - test_label) ** 2)

        print("Loss:", mse / len(test_label))
        print("Time: %.5fms" % ((time.time() - init) * 1000))

        show(test_data, test_label, model, type)
