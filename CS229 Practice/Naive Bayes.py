# A simple naive Bayes model with Laplace smoothing for classfication
# @Time: 3/12/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Naive Bayes.py

import numpy as np
import time
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

data = make_classification ( n_samples = 1200 , n_classes = 2 , n_clusters_per_class = 1 , n_features = 2 , n_redundant = 0 , class_sep = 2 )

train_data = data [ 0 ] [ : 1000 ]
train_label = data [ 1 ] [ : 1000 ]

test_data = data [ 0 ] [ 1000 : ]
test_label = data [ 1 ] [ 1000 : ]

init = time.time ()

phi_j_y_1 = np.ones ( ( len ( train_data [ 0 ] ) , 20 ) )
phi_j_y_0 = np.ones ( ( len ( train_data [ 0 ] ) , 20 ) )

pos = 10
neg = 10

for i in range ( len ( train_label ) ) :
    if train_label [ i ] == 0 :
        neg += 1
        phi_j_y_0 [ 0 ] [ int ( train_data [ i ] [ 0 ] ) + 5 ] += 1
        phi_j_y_0 [ 1 ] [ int ( train_data [ i ] [ 1 ] ) + 5 ] += 1
    else :
        pos += 1
        phi_j_y_1 [ 0 ] [ int ( train_data [ i ] [ 0 ] ) + 5 ] += 1
        phi_j_y_1 [ 1 ] [ int ( train_data [ i ] [ 1 ] ) + 5 ] += 1

phi_j_y_0 /= neg
phi_j_y_1 /= pos
phi_y_0 = (neg - 9) / (neg + pos - 18)
phi_y_1 = (pos - 9) / (pos + neg - 18)

correct = 0

x0_1 = [] ; x0_2 = []
x1_1 = [] ; x1_2 = []

for i in range ( len ( test_label ) ) :
    phi_x_y_1 = phi_j_y_1 [ 0 ] [ int ( test_data [ i ] [ 0 ] ) + 5 ] * phi_j_y_1 [ 1 ] [ int ( test_data [ i ] [ 1 ] ) + 5 ] * phi_y_1
    phi_x_y_0 = phi_j_y_0 [ 0 ] [ int ( test_data [ i ] [ 0 ] ) + 5 ] * phi_j_y_0 [ 1 ] [ int ( test_data [ i ] [ 1 ] ) + 5 ] * phi_y_0

    p_0 = phi_x_y_0 / ( phi_x_y_0 + phi_x_y_1 )
    p_1 = phi_x_y_1 / ( phi_x_y_0 + phi_x_y_1 )

    correct += ( ( p_1 >= p_0 ) == test_label [ i ] )

    if ( p_1 >= p_0 ) :
        x1_1.append ( test_data [ i ] [ 0 ] )
        x1_2.append ( test_data [ i ] [ 1 ] )
    else :
        x0_1.append ( test_data [ i ] [ 0 ] )
        x0_2.append ( test_data [ i ] [ 1 ] )

print ( "Correction: %.2f" % ( correct / len ( test_label ) * 100 ) + "%" )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )

plt.subplot ( 1 , 2 , 1 )

plt.scatter ( x0_1 , x0_2 , s = 30 , c = "red" )
plt.scatter ( x1_1 , x1_2 , s = 30 , c = "blue" )

plt.xlabel ( 'X1' )
plt.ylabel ( 'X2' )

plt.title ( "Naive Bayes" )

x0_1 = [] ; x0_2 = []
x1_1 = [] ; x1_2 = []

for i in range ( len ( test_label ) ) :
    if ( test_label [ i ] ) :
        x1_1.append ( test_data [ i ] [ 0 ] )
        x1_2.append ( test_data [ i ] [ 1 ] )
    else :
        x0_1.append ( test_data [ i ] [ 0 ] )
        x0_2.append ( test_data [ i ] [ 1 ] )

plt.subplot ( 1 , 2 , 2 )

plt.scatter ( x0_1 , x0_2 , s = 30 , c = "red" )
plt.scatter ( x1_1 , x1_2 , s = 30 , c = "blue" )

plt.xlabel ( 'X1' )
plt.ylabel ( 'X2' )
plt.title ( "Real Labels" )

plt.show ()