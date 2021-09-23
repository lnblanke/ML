# Training CNN for handwriting numbers
# @Time: 8/13/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: cnn.py

import mnist
from cnn_class import Conv
from MaxPool import MaxPool
from Softmax import Softmax
import numpy as np

# Get train and test images from MNIST dataset
train_img = mnist.train_images()[:10000]
train_label = mnist.train_labels()[:10000]
test_img = mnist.test_images()[:1000]
test_label = mnist.test_labels()[:1000]

# Create CNNm, maxpool, and softmax layers
conv = Conv(8)
pool = MaxPool()
sm = Softmax(13 * 13 * 8, 10)

# Forward process
def forward(img, label):
    out = conv.forward((img / 255) - 0.5)
    out = pool.forward(out)
    out = sm.forward(out)

    loss = - np.log(out[label])

    accuracy = 0

    if np.argmax(out) == label:
        accuracy = 1

    return out, loss, accuracy

# Train the CNN
def train(img, label, rate = 0.005):
    out, loss, accuracy = forward(img, label)

    grad = np.zeros(10)
    grad[label] = -1 / out[label]

    grad = sm.backprop(grad, rate)
    grad = pool.backprop(grad)
    grad = conv.backprop(grad, rate)

    return loss, accuracy

loss = 0
correct = 0

if __name__ == '__main__':
    for i, (im, label) in enumerate(zip(train_img, train_label)):
        ls, accuracy = train(im, label)

        loss += ls
        correct += accuracy

        if (i + 1) % 100 == 0:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, correct)
            )
            loss = 0
            correct = 0

    loss = 0
    correct = 0

    # Test
    for img, label in zip(test_img, test_label):
        _, l, acc = forward(img, label)

        loss += l
        correct += acc

    num_tests = len(test_img)

    print('Test Loss:', loss / num_tests)
    print('Test Accuracy:', correct / num_tests)
