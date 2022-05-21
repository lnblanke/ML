# A simple Network in Network model on ImageNet classification using tensorflow
# @Time: 6/27/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: NiN.py

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, GlobalAveragePooling2D


def NiN_block(input, kernel_size, filters, stride):
    x = Conv2D(kernel_size = kernel_size, filters = filters, strides = stride, padding = "same", activation = "relu")(
        input)
    x = Conv2D(kernel_size = 1, filters = filters, activation = "relu")(x)
    x = Conv2D(kernel_size = 1, filters = filters, activation = "relu")(x)

    return x


def createNiN():
    input = Input(shape = (224, 224, 3))
    x = NiN_block(input, 11, 96, 4)
    x = MaxPooling2D(3, strides = 2)(x)
    x = NiN_block(x, 5, 256, 1)
    x = MaxPooling2D(3, strides = 2)(x)
    x = NiN_block(x, 3, 384, 1)
    x = MaxPooling2D(3, 2)(x)
    x = Dropout(.5)(x)
    x = NiN_block(x, 3, 10, 1)
    x = GlobalAveragePooling2D()(x)

    model = tf.keras.Model(inputs = input, outputs = x, name = "NiN")

    return model
