# A sample fully connected neural network that learns to perform XOR calculation
# @Time: 12/15/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: multi-class-ann.py

from blocks import Dense
from model import Model
import numpy as np
from tools import get_classification_data, show

rate = 1e-2  # Learning rate
epoch = 100  # Learning epochs
patience = 10  # Early stop patience

model = Model("ANN")
model.add(Dense(2, 8, "relu", name = "Relu-1"))
model.add(Dense(8, 16, "relu", name = "Relu-2"))
model.add(Dense(16, 4, "relu", name = "Relu-3"))
model.add(Dense(4, 3, "softmax", name = "Softmax"))

# Get data
train_x, test_x, train_y, test_y = get_classification_data(samples = 1000, features = 2,
                                                           classes = 3, sep = 1, random_state = 0)

if __name__ == '__main__':
    model.fit(train_x, train_y, epochs = epoch, loss_func = "cross entropy loss", learning_rate = rate,
              patience = patience)

    pred = model.predict(test_x)

    print("Accuracy: %.2f" % (np.sum(pred == test_y) / len(test_y) * 100) + "%")

    show(test_x, pred, test_y, "ANN")
