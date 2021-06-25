# A simple K-means classifier for unsupervised learning
# @Time: 3/28/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: K-means.py

import numpy as np
import time
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

data = make_classification ( n_samples = 1000 , n_classes = 2 , n_clusters_per_class = 1 , n_features = 2 , n_redundant = 0 , class_sep = 2 )

train_data = data [ 0 ] [ : 1000 ]

init = time.time ()

k = 2

p = np.zeros ( ( k , len ( train_data [ 0 ] ) ) )

for i in range ( k ) :
    p [ i ] = train_data [ i ]

color = np.zeros ( len ( train_data ) )

count = 0

_cost = 0

while True :
    for i in range ( len ( train_data ) ) :
        color [ i ] = np.argmin ( [ np.linalg.norm ( train_data [ i ] - p [ 0 ] ) , np.linalg.norm ( train_data [ i ] - p [ 1 ] ) ] )

    for i in range ( k ) :
        up = 0
        down = 0
        for j in range ( len ( train_data ) ) :
            up += ( color [ j ] == i ) * train_data [ j ]
            down += ( color [ j ] == i )

        p [ i ] = up / down

    cost = 0
    count += 1
    for i in range ( len ( train_data ) ) :
        cost += np.linalg.norm ( train_data [ i ] - p [ int ( color [ i ] ) ] )

    print ( "Epoch: %d Loss: %.5f" % ( count , cost ) )

    if cost == _cost :
        break

    _cost = cost

class0_1 = [ ]
class0_2 = [ ]
class1_1 = [ ]
class1_2 = [ ]

for i in range ( len ( train_data ) ) :
    if color [ i ] == 0 :
        class0_1.append ( train_data [ i ] [ 0 ] )
        class0_2.append ( train_data [ i ] [ 1 ] )
    else :
        class1_1.append ( train_data [ i ] [ 0 ] )
        class1_2.append ( train_data [ i ] [ 1 ] )

plt.scatter ( class0_1 , class0_2 , s = 30 , c = "red" )
plt.scatter ( class1_1 , class1_2 , s = 30 , c = "blue" )

plt.xlabel ( 'X1' )
plt.ylabel ( 'X2' )

print ( "Time: %.5fms" % ( ( time.time () - init ) * 1000 ) )

plt.show ()