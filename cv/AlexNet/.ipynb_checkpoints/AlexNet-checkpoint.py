# A simple AlexNet model on ImageNet classification using Tensorflow
# @Time: 6/25/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: AlexNet.py

import tensorflow as tf
from tensorflow.keras.layers import *

def createAlexNet():
    model = tf.keras.Sequential([
        Conv2D(input_shape = (224, 224, 3), kernel_size = (11, 11), filters = 96, strides = 4, padding = "valid",
            activation = "relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size = (3, 3), strides = 2),
        Conv2D(kernel_size = (5, 5), filters = 256, strides = 4, padding = "same", activation = "relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size = (3, 3), strides = 2),
        Conv2D(kernel_size = (3, 3), filters = 384, padding = "same", activation = "relu"),
        Conv2D(kernel_size = (3, 3), filters = 384, padding = "same", activation = "relu"),
        Conv2D(kernel_size = (3, 3), filters = 256, padding = "same", activation = "relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size = (3, 3), strides = 2),
        Dropout(rate = .5),
        Activation(activation = "relu"),
        Dropout(rate = .5),
        Activation(activation = "relu"),
        Dense(1000, activation = "softmax")
    ])

    return model