# Get and normalize data from datasets
# @Time: 5/21/22
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: data.py

import tensorflow as tf


def getCifar(batch_size, train_size = .8):
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

    train_x = train_x / 255.0
    test_x = test_x / 255.0

    train_size = int(len(train_y) * train_size)

    train = tf.data.Dataset.from_tensor_slices((train_x[:train_size],
                                                train_y[:train_size])).shuffle(train_size).batch(batch_size).prefetch(1)
    val = tf.data.Dataset.from_tensor_slices((train_x[train_size:],
                                              train_y[train_size:])).batch(batch_size)
    test = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size)

    return train, val, test
