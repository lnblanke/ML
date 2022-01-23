# A sample fully connected neural network that learns to perform XOR calculation
# @Time: 8/13/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: ann.py

from blocks import Dense
from model import Model
import numpy as np
from tools import get_classification_data, show

rate = 1e-2  # Learning rate
epoch = 500  # Learning epochs
patience = 10  # Early stop patience

model = Model("ANN")
model.add(Dense(2, 16, "relu"))
model.add(Dense(16, 4, "relu"))
model.add(Dense(4, 1, "sigmoid"))

# Get data
train_x, test_x, train_y, test_y = get_classification_data(samples = 1000, features = 2,
                                                           classes = 2, sep = 1)

if __name__ == '__main__':
    model.fit(train_x, train_y, epochs = epoch, loss_func = "mse", learning_rate = rate, patience = patience)

    pred = model.predict(test_x)

    print("Accuracy: %.2f" % (np.sum(pred == test_y) / len(test_y) * 100) + "%")

    show(test_x, pred, test_y, "ANN")
