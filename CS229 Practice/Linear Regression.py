# Some simple linear regression models
# @Time: 3/1/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Linear Regression.py

from sklearn.datasets import make_regression
import numpy as np
import time
import matplotlib.pyplot as plt

data = make_regression ( n_samples = 50 , n_features = 2 , bias = 1 )

train_data = data [ 0 ] [ : 40 ]
train_data = np.c_ [ train_data , np.ones ( len ( train_data ) ) ]
train_label = data [ 1 ] [ : 40 ]

test_data = data [ 0 ] [ 40 : ]
test_data = np.c_ [ test_data , np.ones ( len ( test_data ) ) ]
test_label = data [ 1 ] [ 40 : ]

# BGD
init = time.time()

rate = 0.1

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

print ( "Loss:" , mse / len ( test_label ) )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )

plt.scatter ( train_data [ : , 0 ] , train_data [ : , 1 ] , c = "black" )
plt.xlabel ( 'X1' )
plt.ylabel ( 'X2' )

x = np.arange ( -5 , 5 )
y = -1 * ( weight [ 2 ] + weight [ 0 ] * x ) / weight [ 1 ]
plt.plot ( x , y )

plt.title ( "BGD" )

plt.show ()

# SGD
init = time.time()

rate = .1

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

print ( "Loss:" , mse / len ( test_label ) )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )

plt.scatter ( train_data [ : , 0 ] , train_data [ : , 1 ] , c = "black" )
plt.xlabel ( 'X1' )
plt.ylabel ( 'X2' )

x = np.arange ( -5 , 5 )
y = -1 * ( weight [ 2 ] + weight [ 0 ] * x ) / weight [ 1 ]
plt.plot ( x , y )

plt.title ( "SGD" )

plt.show ()

# Normal Equation
init = time.time()

weight = np.dot ( np.dot ( np.linalg.inv ( np.dot ( train_data.transpose() , train_data ) ) , train_data.transpose() ) , train_label )

mse = np.sum ( ( np.dot ( test_data , weight ) - test_label ) ** 2 )

print ( "Normal Equation:" )
print ( "Loss:" , mse / len ( test_label ) )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )

plt.scatter ( train_data [ : , 0 ] , train_data [ : , 1 ] , c = "black" )
plt.xlabel ( 'X1' )
plt.ylabel ( 'X2' )

x = np.arange ( -5 , 5 )
y = -1 * ( weight [ 2 ] + weight [ 0 ] * x ) / weight [ 1 ]
plt.plot ( x , y )

plt.title ( "Normal Equation" )

plt.show ()

# Locally Weighted Regression
init = time.time()

mse = 0

print ( "Locally Weighted Regression:" )

plt.scatter ( train_data [ : , 0 ] , train_data [ : , 1 ] , c = "black" )
plt.xlabel ( 'X1' )
plt.ylabel ( 'X2' )

plt.title ( "Normal Equation" )

for i in range ( len ( test_label ) ) :
    cost = 1e7

    rate = .1

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

    x = np.arange ( -5 , 5 )
    y = -1 * (weight [ 2 ] + weight [ 0 ] * x) / weight [ 1 ]
    plt.plot ( x , y )

    mse += ( np.dot ( test_data [ i ] , weight ) - test_label [ i ] ) ** 2

print ( "Loss:" , mse / len ( test_label ) )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )

plt.title ( "LWR" )
plt.show ()