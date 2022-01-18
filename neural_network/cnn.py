# A sample conv neural network that learns to recognize MNIST dataset
# @Time: 8/13/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: cnn.py

import numpy as np
import tensorflow as tf
from blocks import Conv, Dense, MaxPool
from model import Model

# Get train and test images from MNIST dataset
(train_img, train_label), (test_img, test_label) = tf.keras.datasets.mnist.load_data()

rate = 1e-2  # Learning rate
epoch = 30  # Learning epochs
patience = 10  # Early stop patience

train_img = train_img / 255 - .5
test_img = test_img / 255 - .5

model = Model("CNN")
model.add(Conv(3, 8, "valid"))
model.add(MaxPool(2))
model.add(Dense(13 * 13 * 8, 10, "softmax"))

if __name__ == '__main__':
    model.fit(train_img, train_label, loss_func = "cross entropy loss", epochs = epoch, learning_rate = rate,
              patience = patience)

    pred = model.predict(test_img)

    print("Accuracy: %.2f" % (np.sum(pred == test_label) / len(test_label) * 100) + "%")
