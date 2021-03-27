# A simple logistic regression model for classfication
# @Time: 3/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Logistic Regression.py

import numpy as np
import time
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

data = make_classification ( n_samples = 1200 , n_classes = 2 , n_clusters_per_class = 1 , n_features = 2 , n_redundant = 0 , class_sep =  2 )

train_data = data [ 0 ] [ : 1000 ]
train_data = np.c_ [ train_data , np.ones ( len ( train_data ) ) ]
train_label = data [ 1 ] [ : 1000 ]

test_data = data [ 0 ] [ 1000 : ]
test_data = np.c_ [ test_data , np.ones ( len ( test_data ) ) ]
test_label = data [ 1 ] [ 1000 : ]

def sigmoid ( num ) :
    return 1.0 / ( 1 + np.exp ( -1 * num ) )

init = time.time ()

cost = 1e7

rate = .00005

weight = np.random.rand ( len ( train_data [ 0 ] ) )

count = 0

while True :
    gradient = 0

    for i in range ( len ( train_label ) ) :
        gradient += ( train_label [ i ] - sigmoid ( np.dot ( train_data [ i ] , weight ) ) ) * train_data [ i ]

    weight += rate * gradient
    result = sigmoid ( np.dot ( train_data , weight ) )
    correct = 0

    for i in range ( len ( train_label ) ) :
        if ( result [ i ] >= 0.5 ) == train_label [ i ] :
            correct += 1

    count += 1

    print ( "Epoch: %d Accuracy: %.5f" % (count , correct / len ( train_label )) )

    if correct / len ( train_label ) > .95 :
        break

result = sigmoid ( np.dot ( test_data , weight ) )
correct = 0

x0_1 = [] ; x0_2 = []
x1_1 = [] ; x1_2 = []

for i in range ( len ( test_label ) ) :
    if ( result [ i ] >= 0.5 ) == test_label [ i ] :
        correct += 1

    if ( test_label [ i ] ) :
        x1_1.append ( test_data [ i ] [ 0 ] )
        x1_2.append ( test_data [ i ] [ 1 ] )
    else :
        x0_1.append ( test_data [ i ] [ 0 ] )
        x0_2.append ( test_data [ i ] [ 1 ] )

plt.scatter ( x0_1 , x0_2 , s = 30 , c = "red" )
plt.scatter ( x1_1 , x1_2 , s = 30 , c = "blue" )

plt.xlabel ( 'X1' )
plt.ylabel ( 'X2' )

x = np.arange ( -20 , 20 )
y = -1 * ( weight [ 2 ] + weight [ 0 ] * x ) / weight [ 1 ]
plt.plot ( x , y )

print ( "Weight:" , weight )
print ( "Correction: %.2f" % ( correct / len ( test_label ) * 100 ) + "%" )
print ( "Time: %.5fms" % ( ( time.time () - init ) * 1000 ) )

plt.show ()