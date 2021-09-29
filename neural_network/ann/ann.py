# Testing whether a person is male or female with weight and height using common neural network
# @Time: 8/13/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: ann.py

import numpy as np
from neural_network.function import *

# Weights
w1 = np.random.normal()
w2 = np.random.normal()
w3 = np.random.normal()
w4 = np.random.normal()
w5 = np.random.normal()
w6 = np.random.normal()

# Biases
b1 = np.random.normal()
b2 = np.random.normal()
b3 = np.random.normal()

# Get value for the first hidden layer and the output
# x: input, 0: weight, 1: height
def feedForward(x):
    h1 = sigmoid(x[0] * w1 + x[1] * w2 + b1)
    h2 = sigmoid(x[0] * w3 + x[1] * w4 + b2)
    o = sigmoid(h1 * w5 + h2 * w6 + b3)

    return o

n = 0.1  # Learning rate
epoch = 1000  # Learning epochs

# Input data
data = np.array([[-2, -1], [25, 6], [17, 4], [-15, -6]])
y = np.array([1, 0, 0, 1])

if __name__ == '__main__':
    # Learn
    for i in range(epoch):
        for j in range(len(y)):
            h1 = sigmoid(data[j][0] * w1 + data[j][1] * w2 + b1)
            h2 = sigmoid(data[j][0] * w3 + data[j][1] * w4 + b2)
            o = sigmoid(h1 * w5 + h2 * w6 + b3)

            # Find stochastic gradient descent
            dL_dpred = -2 * (y[j] - o)
            dpred_dh1 = w5 * dsigmoid(o)
            dpred_dh2 = w6 * dsigmoid(o)
            dh1_dw1 = data[j][0] * dsigmoid(h1)
            dh1_dw2 = data[j][1] * dsigmoid(h1)
            dh2_dw3 = data[j][0] * dsigmoid(h2)
            dh2_dw4 = data[j][1] * dsigmoid(h2)
            dpred_dw5 = h1 * dsigmoid(o)
            dpred_dw6 = h2 * dsigmoid(o)
            dh1_db1 = dsigmoid(h1)
            dh2_db2 = dsigmoid(h2)
            dpred_db3 = dsigmoid(o)

            # Adjust weights and biases
            w1 -= n * dL_dpred * dpred_dh1 * dh1_dw1
            w2 -= n * dL_dpred * dpred_dh1 * dh1_dw2
            w3 -= n * dL_dpred * dpred_dh2 * dh2_dw3
            w4 -= n * dL_dpred * dpred_dh2 * dh2_dw4
            w5 -= n * dL_dpred * dpred_dw5
            w6 -= n * dL_dpred * dpred_dw6
            b1 -= n * dL_dpred * dpred_dh1 * dh1_db1
            b2 -= n * dL_dpred * dpred_dh1 * dh2_db2
            b3 -= n * dL_dpred * dpred_db3

            # Print learning results
            if i % 10 == 0:
                y_preds = np.apply_along_axis(feedForward, 1, data)
                loss = mse(y, y_preds)
                print("Epoch %d loss: %.3f" % (i, loss))
