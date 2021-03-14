# Some simple linear regression models
# @Time: 3/1/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Linear Regression.py

import csv
import numpy as np
import time
import pandas as pd

train = pd.read_csv ( "regression_train.csv" )
test = pd.read_csv ( "regression_test.csv" )
train [ "c" ] = np.ones ( len ( train ) )
test [ "c" ] = np.ones ( len ( test ) )

train_data = train [ [ "x1" , "x2" , "x3" , "c" ] ].values
train_label = train [ "y" ].values
test_data = test [ [ "x1" , "x2" , "x3" , "c" ] ].values
test_label = test [ "y" ].values

# BGD
init = time.time()

rate = 0.000001

weight = np.random.rand ( len ( train_data [ 0 ] ) )

count = 0

print ( "Batch Gadient Descent:" )

while True :
    gradient = np.zeros ( len ( train_data [ 0 ] ) )

    for i in range ( len ( train_data ) ) :
        gradient += ( np.dot ( train_data [ i ] , weight ) - train_label [ i ] ) * train_data [ i ]

    weight -= rate / len ( train_label ) * gradient

    cost = 0.5 * np.sum ( ( np.dot ( train_data , weight ) - train_label ) ** 2 )

    count += 1

    print ( "Epoch: %d Loss: %.5f" % (count , cost) )

    if cost < 650 :
        break

mse = np.sum ( ( np.dot ( test_data , weight ) - test_label ) ** 2 )

print ( "Weight:" , weight )
print ( "Loss:" , mse / len ( test_label ) )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )

# SGD
init = time.time()

rate = .0000005

weight = np.random.rand ( len ( train_data [ 0 ] ) )

count = 0

print ( "Stochastic Gradient Descent:" )

while True :
    index = int ( np.random.rand() * len ( train_label ) )
    gradient = ( np.dot ( train_data [ index ] , weight ) - train_label [ index ] ) * train_data [ index ]

    weight -= rate * gradient

    cost = 0.5 * np.sum ( ( np.dot ( train_data , weight ) - train_label ) ** 2 )

    count += 1

    print ( "Epoch: %d Loss: %.5f" % (count , cost) )
    if cost < 650 :
        break

mse = np.sum ( ( np.dot ( test_data , weight ) - test_label ) ** 2 )

print ( "Weight:" , weight )
print ( "Loss:" , mse / len ( test_label ) )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )

# Normal Equation
init = time.time()

weight = np.dot ( np.dot ( np.linalg.inv ( np.dot ( train_data.transpose() , train_data ) ) , train_data.transpose() ) , train_label )

mse = np.sum ( ( np.dot ( test_data , weight ) - test_label ) ** 2 )

print ( "Normal Equation:" )
print ( "Weight:" , weight )
print ( "Loss:" , mse / len ( test_label ) )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )

# Locally Weighted Regression
init = time.time()

mse = 0

print ( "Locally Weighted Regression:" )

for i in range ( len ( test_label ) ) :
    cost = 1e7

    rate = .0000005

    weight = np.random.rand ( len ( train_data [ 0 ] ) )

    loss = 1e10

    bias = np.empty ( len ( train_data ) )

    for j in range ( len ( train_data ) ) :
        bias [ j ] = np.exp ( -1 * np.linalg.norm ( train_label [ j ] - test_label [ i ] ) ** 2 / 1e5 )

    while True :
        for j in range ( len ( train_data ) ) :
            gradient = ( np.dot ( train_data [ j ] , weight ) - train_label [ j ] ) * train_data [ j ]

            weight -= rate * bias [ j ] * gradient

        cost = np.abs ( np.dot ( train_data , weight ) - train_label )

        for j in range ( len ( train_data ) ) :
            cost [ j ] *= bias [ j ]

        if np.max ( cost ) < 10 :
            break

    print ( "Weight for test data #%d:" % ( i + 1 ) ,  weight )

    mse += ( np.dot ( test_data [ i ] , weight ) - test_label [ i ] ) ** 2

print ( "Loss:" , mse / len ( test_label ) )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )