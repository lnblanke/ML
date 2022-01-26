# A sample recursive neural network for text classification
# @Time: 8/13/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: cnn.py

import numpy as np
import tensorflow as tf
from blocks import RNN, Dense
from model import Model
import os

path = os.path.join("glove.6B.100d.txt")

embedding_indices = {}

with open(path) as f:
    for line in f:
        word, coef = line.split(maxsplit = 1)
        coef = np.fromstring(coef, "f", sep = " ")

        embedding_indices[word] = coef


def embedding(x):
    word_idx = tf.keras.datasets.imdb.get_word_index()
    embedding_dim = 100

    l, w = x.shape
    embed = np.zeros((l, w, embedding_dim))

    vec_to_word = {vec + 3: ww for ww, vec in word_idx.items()}
    vec_to_word[0] = "<pad>"
    vec_to_word[1] = "<sos>"
    vec_to_word[2] = "<unk>"

    for i in range(l):
        for j in range(w):
            embedding_vec = embedding_indices.get(vec_to_word[x[i][j]])

            if embedding_vec is not None:
                embed[i][j] = embedding_vec

    return embed


word_size = 15000

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.imdb.load_data(num_words = word_size)

max_len = 300
train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, max_len)[:1000]
train_y = train_y[:1000]
test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x, max_len)[:200]
test_y = test_y[:200]

train_x_embed = embedding(train_x)
test_x_embed = embedding(test_x)

rate = 1e-2  # Learning rate
epoch = 100  # Learning epochs
patience = 10  # Early stop patience

model = Model("RNN")
model.add(RNN(input_size = 100, output_size = 64, units = 128))
model.add(Dense(64, 2, activation = "softmax"))

if __name__ == '__main__':
    model.fit(train_x_embed, train_y, loss_func = "cross entropy loss", epochs = epoch, learning_rate = rate,
              patience = patience)

    pred = model.predict(test_x_embed)

    print("Accuracy: %.2f" % (np.sum(pred == test_y) / len(test_y) * 100) + "%")
