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

rate = 0.5

init = time.time()

count = 0

while True :
    predict = []

    gradient = np.zeros ( ( 10 , 28 * 28 ) )

    for i in range ( len ( train_label ) ) :
        predict = np.exp ( np.dot ( weight , train_img [ i ] ) ) / np.sum ( np.exp ( np.dot ( weight , train_img [ i ] ) ) )

        for j in range ( 10 ) :
            gradient [ j ] += train_img [ i ] * ( ( train_label [ i ] == j ) - predict [ j ] )

    # plt.imshow ( np.reshape ( train_img [ len ( train_label ) - 1] , ( 28 , 28 ) ) )
    # pylab.show ()

    weight -= rate * -1 * gradient / len ( train_label )

    cost = 0

    for i in range ( len ( train_label ) ) :
        predict = np.exp ( np.dot ( weight , train_img [ i ] ) ) / np.sum ( np.exp ( np.dot ( weight , train_img [ i ] ) ) )

        for j in range ( 10 ) :
            cost += ( train_label [ i ] == j ) * np.log ( predict [ j ] )

    cost = -1 * cost / len ( train_label )

    count += 1

    print ( "Epoch: %d Loss: %.5f" % ( count , cost ) )

    if cost < .1 :
        break

correct = 0

for i in range ( len ( test_img ) ) :
    if test_label [ i ] == np.argmax ( np.dot ( weight , test_img [ i ] ) ) :
        correct += 1

print ( "Correction: %.2f" % ( correct / len ( test_label ) * 100 ) + "%" )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )