# A sample recursive neural network for text classification
# @Time: 8/13/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: cnn.py

import numpy as np
import tensorflow as tf
from blocks import RNN, Dense
from model import Model

word_size = 15000

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.imdb.load_data(num_words = word_size)

max_len = 300
train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, max_len)[:5000]
train_y = train_y[:5000]
test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x, max_len)[:1000]
test_y = test_y[:5000]

rate = 1e-2  # Learning rate
epoch = 30  # Learning epochs
patience = 10  # Early stop patience

model = Model("RNN")
model.add(RNN(input_size = 1, output_size = 8, units = 16))
model.add(Dense(8, 2, "softmax"))

if __name__ == '__main__':
    model.fit(np.expand_dims(train_x, axis = -1), train_y, loss_func = "cross entropy loss", epochs = epoch,
              learning_rate = rate,
              patience = patience)

    pred = model.predict(np.expand_dims(test_x, axis = -1))

    print("Accuracy: %.2f" % (np.sum(pred == test_y) / len(test_y) * 100) + "%")
