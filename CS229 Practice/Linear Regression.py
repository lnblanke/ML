# A simple linear regression model
# @Time: 3/1/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Linear Regression.py

import csv
import numpy as np
import time

train = csv.reader ( open ( "regression_train.csv" ) )
test = csv.reader ( open ( "regression_test.csv" ) )

train_data = np.empty ( ( 100 , 4 ) )
train_label = np.empty ( 100 )
test_data = np.empty ( ( 20 , 4 ) )
test_label = np.empty ( 20 )

for line in train :
    if line [ 0 ] == "#" :
        continue

    train_data [ int ( line [ 0 ] ) - 1 ] [ 0 ] = line [ 1 ]
    train_data [ int ( line [ 0 ] ) - 1 ] [ 1 ] = line [ 2 ]
    train_data [ int ( line [ 0 ] ) - 1 ] [ 2 ] = line [ 3 ]
    train_data [ int ( line [ 0 ] ) - 1 ] [ 3 ] = 1

    train_label [ int ( line [ 0 ] ) - 1 ] = line [ 4 ]

for line in test :
    if line [ 0 ] == "#" :
        continue

    test_data [ int ( line [ 0 ] ) - 1 ] [ 0 ] = line [ 1 ]
    test_data [ int ( line [ 0 ] ) - 1 ] [ 1 ] = line [ 2 ]
    test_data [ int ( line [ 0 ] ) - 1 ] [ 2 ] = line [ 3 ]
    test_data [ int ( line [ 0 ] ) - 1 ] [ 3 ] = 1

    test_label [ int ( line [ 0 ] ) - 1 ] = line [ 4 ]

# BGD
init = time.time()

cost = 1e7

rate = 0.000001

weight = np.random.rand ( 4 )

while True :
    for i in range ( 0 , len ( train_data ) ) :
        gradient = ( np.dot ( train_data [ i ] , weight ) - train_label [ i ] ) * train_data [ i ]

        weight -= rate * gradient

    if np.max ( np.abs ( np.dot ( train_data , weight ) - train_label ) ) < 20 :
        break

mse = np.sum ( ( np.dot ( test_data , weight ) - test_label ) ** 2 )

print ( "Batch Gadient Descent:" )
print ( "Weight:" , weight )
print ( "Loss:" , mse / len ( test_label ) )
print ( "Time: %.5fms" % ( time.time() - init ) )

# SGD
init = time.time()

cost = 1e7

rate = .0000005

weight = np.random.rand ( 4 )

loss = 1e10

while True :
    index = int ( np.random.rand() * len ( train_label ) )
    gradient = ( np.dot ( train_data [ index ] , weight ) - train_label [ index ] ) * train_data [ index ]

    weight -= rate * gradient

    if np.max ( np.abs ( np.dot ( train_data , weight ) - train_label ) ) < 10 :
        break

mse = np.sum ( ( np.dot ( test_data , weight ) - test_label ) ** 2 )

print ( "Stochastic Gradient Descent:" )
print ( "Weight:" , weight )
print ( "Loss:" , mse / len ( test_label ) )
print ( "Time: %.5fms" % ( time.time() - init ) )

# Normal Equation
init = time.time()

weight = np.dot ( np.dot ( np.linalg.inv ( np.dot ( train_data.transpose() , train_data ) ) , train_data.transpose() ) , train_label )

mse = np.sum ( ( np.dot ( test_data , weight ) - test_label ) ** 2 )

print ( "Normal Equation:" )
print ( "Weight:" , weight )
print ( "Loss:" , mse / len ( test_label ) )
print ( "Time: %.5fms" % ( time.time() - init ) )

# Locally Weighted Regression
init = time.time()

mse = 0

print ( "Locally Weighted Regression:" )

for i in range ( len ( test_label ) ) :
    cost = 1e7

    rate = .0000005

    weight = np.random.rand ( 4 )

    loss = 1e10

    bias = np.empty ( 100 )

    for j in range ( 100 ) :
        bias [ j ] = np.exp ( -1 * np.linalg.norm ( train_label [ j ] - test_label [ i ] ) ** 2 )

    while True :
        for j in range ( 0 , len ( train_data ) ) :
            gradient = ( np.dot ( train_data [ j ] , weight ) - train_label [ j ] ) * train_data [ j ]

            weight -= rate * gradient

        if np.max ( np.abs ( np.dot ( train_data , weight ) - train_label ) ) < 20 :
            break

    print ( "Weight for test data #%d:" % ( i ) ,  weight )

    mse += ( np.dot ( test_data [ i ] , weight ) - test_label [ i ] ) ** 2

print ( "Loss:" , mse / len ( test_label ) )
print ( "Time: %.5fms" % ( time.time() - init ) )