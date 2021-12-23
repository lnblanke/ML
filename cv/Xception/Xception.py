# A simple Xception model on ImageNet classification using Tensorflow
# @Time: 7/4/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Xception.py

import tensorflow as tf
from tensorflow.keras.layers import *

def entry(input, filter):
    x = Activation(activation = "relu")(input)
    x = SeparableConv2D(kernel_size = 3, filters = filter, padding = "same", activation = "relu")(x)
    x = SeparableConv2D(kernel_size = 3, filters = filter, padding = "same")(x)
    x = MaxPooling2D(pool_size = 3, strides = 2, padding = "same")(x)
    input = Conv2D(kernel_size = 1, filters = filter, padding = "same", strides = 2)(input)
    x = Concatenate()([x, input])

    return x

def middle(input):
    x = None

    for i in range(8):
        x = Activation(activation = "relu")(input)
        x = SeparableConv2D(kernel_size = 3, filters = 728, padding = "same", activation = "relu")(x)
        x = SeparableConv2D(kernel_size = 3, filters = 728, padding = "same", activation = "relu")(x)
        x = SeparableConv2D(kernel_size = 3, filters = 728, padding = "same")(x)
        x = Concatenate()([x, input])

    return x

def createXception():
    input = Input(shape = (299, 299, 3))
    x = Conv2D(kernel_size = 3, strides = 2, filters = 32, activation = "relu", padding = "same")(input)
    x = Conv2D(kernel_size = 3, filters = 64, padding = "same")(x)
    x = entry(x, 128)
    x = entry(x, 256)
    x = entry(x, 728)
    x = middle(x)
    concat = x
    x = Activation(activation = "relu")(x)
    x = SeparableConv2D(kernel_size = 3, filters = 728, padding = "same", activation = "relu")(x)
    x = SeparableConv2D(kernel_size = 3, filters = 1024, padding = "same")(x)
    x = MaxPooling2D(pool_size = 3, strides = 2, padding = "same")(x)
    concat = Conv2D(kernel_size = 1, filters = 1024, padding = "same", strides = 2)(concat)
    x = Concatenate()([x, concat])
    x = SeparableConv2D(kernel_size = 3, filters = 1536, padding = "same", activation = "relu")(x)
    x = SeparableConv2D(kernel_size = 3, filters = 2048, padding = "same", activation = "relu")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation = "relu")(x)
    x = Dropout(.5)(x)
    x = Dense(1000, activation = "sigmoid")(x)

    model = tf.keras.Model(inputs = input, outputs = x, name = "Xception")

    return model

if __name__ == '__main__':
    print(createXception().summary())
    tf.keras.utils.plot_model(createXception(), to_file = "Xception.png", show_shapes = False, show_layer_names = True,
        rankdir = "TB")
