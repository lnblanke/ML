# Visualization tool for training and testing
# @Time: 5/21/22
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: visualization.py.py

import matplotlib.pyplot as plt
import tensorflow as tf


def show_training_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    for i, (acc, val_acc) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'])):
        if (i + 1) % 10 == 0:
            plt.annotate("{:.2f}".format(acc), xy = (i + 1, acc))
            plt.annotate("{:.2f}".format(val_acc), xy = (i + 1, val_acc))

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.show()


def show_training_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    for i, (acc, val_acc) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
        if (i + 1) % 10 == 0:
            plt.annotate("{:.2f}".format(acc), xy = (i + 1, acc))
            plt.annotate("{:.2f}".format(val_acc), xy = (i + 1, val_acc))

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.show()


def show_Cifar_validation_result(model, test_ds):
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    print("Test Accuracy: {:.2%}".format(model.evaluate(test_ds)[1]))

    fig = plt.figure(figsize = (10, 40))
    for sample_data, sample_label in test_ds.take(1):
        pred = tf.argmax(model.predict(sample_data), axis = 1)

        for i, (img, label) in enumerate(zip(sample_data[:9], sample_label[:9])):
            ax = fig.add_subplot(911 + i)
            ax.imshow(img)

            ax.set_title("Labelled as " + labels[int(label)] + ", classified as " + labels[int(pred[i])])
