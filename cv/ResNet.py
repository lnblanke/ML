# A simple ResNet model on ImageNet classification using tensorflow
# @Time: 6/27/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: ResNet.py

import tensorflow as tf
from tensorflow.keras.layers import *

def bottleneck(input, f1, f3, stride):
    x = Conv2D(kernel_size = 1, filters = f1, padding = "same", activation = "relu", strides = stride)(input)
    x = Conv2D(kernel_size = 3, filters = f1, padding = "same", activation = "relu")(x)
    x = Conv2D(kernel_size = 1, filters = f3, padding = "same")(x)

    if stride == 2:
        input = Conv2D(kernel_size = 1, strides = 2, filters = f3, padding = "valid", activation = "relu")(input)

    x = Concatenate()([input, x])
    x = Activation(activation = "relu")(x)

    return x

def block(input, f1, stride):
    x = Conv2D(kernel_size = 3, filters = f1, padding = "same", activation = "relu", strides = stride)(input)
    x = Conv2D(kernel_size = 3, filters = f1, padding = "same", activation = "relu")(x)

    if stride == 2:
        input = Conv2D(kernel_size = 1, strides = 2, filters = f1, padding = "valid", activation = "relu")(input)

    x = Concatenate()([input, x])
    x = Activation(activation = "relu")(x)

    return x

def makeLayers(input, f1, f3, layers, bottleNeck = False):
    if bottleNeck:
        x = bottleneck(input, f1, f3, 2 if f1 != 64 else 1)

        for i in range(layers - 1):
            x = bottleneck(x, f1, f3, 1)
    else:
        x = block(input, f1, 2 if f1 != 64 else 1)

        for i in range(layers - 1):
            x = block(x, f1, 1)

    return x

def createResNet(type):
    if type == 18:
        params = [2, 2, 2, 2]
    elif type == 34:
        params = [3, 4, 6, 3]
    elif type == 50:
        params = [3, 4, 6, 3]
    elif type == 101:
        params = [3, 4, 23, 3]
    elif type == 152:
        params = [3, 8, 36, 3]
    else:
        raise Exception("The parameter is not valid!")

    input = Input(shape = (224, 224, 3))
    x = Conv2D(kernel_size = 7, filters = 64, strides = 2, activation = "relu")(input)
    x = MaxPooling2D(pool_size = 3, strides = 2)(x)

    x = makeLayers(x, 64, 256, params[0], bottleNeck = type >= 50)
    x = makeLayers(x, 128, 512, params[1], bottleNeck = type >= 50)
    x = makeLayers(x, 256, 1024, params[2], bottleNeck = type >= 50)
    x = makeLayers(x, 512, 2048, params[3], bottleNeck = type >= 50)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation = "softmax")(x)

    model = tf.keras.Model(inputs = input, outputs = x, name = "ResNet")

    return model

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.cifar10.load_data()

    model = createResNet(18)

    model.compile(optimizer = "adam", loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = ["accuracy"])

    model.fit(train_data, train_label, epochs = 50, batch_size = 100)

    pred = np.argmax(model.predict(test_data), axis = 1)

    correct = 0

    for i in range(len(test_label)):
        correct += (pred[i] == test_label[i])

    print("Correction: %.2f%%" % (correct / len(test_label) * 100))

    fig = plt.figure(figsize = (10, 40))

    for i in range(9):
        ax = fig.add_subplot(911 + i)
        ax.imshow(test_data[i])

        ax.set_title(
            "Labelled as " + labels[int(test_label[i])] + ", classified as " + labels[int(pred[i])])
