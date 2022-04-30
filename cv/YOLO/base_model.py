import tensorflow as tf
from tensorflow.keras.layers import *


def conv(input, kernel_size, filters, strides, kernel_regularizer, dropout_rate = 0, padding = "same"):
    x = Conv2D(kernel_size = kernel_size, filters = filters, strides = strides, 
               padding = padding, kernel_regularizer = kernel_regularizer)(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(.1)(x)
    
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
        
    return x

def base_model(weight_decay, dropout_rate):
    input = Input(shape = (448, 448, 3))
    
    x = conv(input, 7, 64, 2, l2(weight_decay))
    x = MaxPooling2D(pool_size = 2, strides = 2)(x)
    
    x = conv(x, 3, 192, 1, l2(weight_decay))
    x = MaxPooling2D(pool_size = 2, strides = 2)(x)
    
    x = conv(x, 1, 128, 1, l2(weight_decay), dropout_rate = dropout_rate)
    x = conv(x, 3, 256, 1, l2(weight_decay), dropout_rate = dropout_rate)
    x = conv(x, 1, 256, 1, l2(weight_decay), dropout_rate = dropout_rate)
    x = conv(x, 3, 512, 1, l2(weight_decay))
    x = MaxPooling2D(pool_size = 2, strides = 2)(x)
    
    for _ in range(4):
        x = conv(x, 1, 256, 1, l2(weight_decay), dropout_rate = dropout_rate)
        x = conv(x, 3, 512, 1, l2(weight_decay), dropout_rate = dropout_rate)
        
    x = conv(x, 1, 512, 1, l2(weight_decay), dropout_rate = dropout_rate)
    x = conv(x, 3, 1024, 1, l2(weight_decay))
    
    for _ in range(2):
        x = conv(x, 1, 512, 1, l2(weight_decay), dropout_rate = dropout_rate)
        x = conv(x, 3, 1024, 1, l2(weight_decay), dropout_rate = dropout_rate)

    x = conv(x, 3, 1024, 1, l2(weight_decay), dropout_rate = dropout_rate)
    x = conv(x, 3, 1024, 2, l2(weight_decay))
    
    x = conv(x, 3, 1024, 1, l2(weight_decay), dropout_rate = dropout_rate)
    x = conv(x, 3, 1024, 1, l2(weight_decay), dropout_rate = dropout_rate)
    
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    model = tf.keras.Model(inputs = input, outputs = x, name = "base")

    return model