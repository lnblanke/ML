# A simple softmax regression model predicting MNIST dataset
# @Time: 3/9/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Softmax Regression.py

import numpy as np
from tensorflow.python.keras.datasets import mnist
import matplotlib.pyplot as plt
import pylab
import time

( train_img , train_label ) , ( test_img , test_label ) = mnist.load_data ()

train_img = np.reshape ( train_img [ :1000 ] / 255 - .5 , ( 1000 , 28 * 28 ) )
test_img = np.reshape ( test_img [ :100 ] / 255 - .5 , ( 100 , 28 * 28 ) )

train_label = train_label [ :1000 ]
test_label = test_label [ :100 ]

weight = np.random.random ( ( 10 , 28 * 28 ) )

rate = 0.1

init = time.time()

count = 0

while True :
    gradient = np.zeros ( ( 10 , 28 * 28 ) )
    cost = 0

    for i in range ( len ( train_label ) ) :
        pred = np.zeros ( 10 )

        for j in range ( 10 ) :
            pred [ j ] = np.exp ( np.dot ( weight [ j ] , train_img [ i ] ) )

        for j in range ( 10 ) :
            cost += -1 * ( train_label [ i ] == j ) * np.log ( pred [ j ] / np.sum ( pred ) )
            gradient [ j ] += train_img [ i ] * ( ( train_label [ i ] == j ) - pred [ j ] / np.sum ( pred ) )

    weight -= -1 / len ( train_label ) * rate * gradient

    count += 1

    print ( "Epoch: %d Loss: %.5f" % ( count , cost ) )

    if cost < 500 :
        break

correct = 0

for i in range ( len ( test_img ) ) :
    if test_label [ i ] == np.argmax ( np.dot ( weight , test_img [ i ] ) ) :
        correct += 1

print ( "Correction: %.2f" % ( correct / len ( test_label ) * 100 ) + "%" )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )