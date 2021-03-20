# A simple naive Bayes model with Laplace smoothing for classfication
# @Time: 3/12/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Naive Bayes.py

import numpy as np
import time
import pandas as pd

train = pd.read_csv ( "classification_train.csv" )
test = pd.read_csv ( "classification_test.csv" )

train_data = train [ [ "x1" , "x2" , "x3" ] ].values
train_label = train [ "y" ].values
test_data = test [ [ "x1" , "x2" , "x3" ] ].values
test_label = test [ "y" ].values

init = time.time ()

phi_j_y_1 = np.ones ( (3 , 10) )
phi_j_y_0 = np.ones ( (3 , 10) )

pos = 10
neg = 10

for i in range ( len ( train_label ) ) :
    if train_label [ i ] == 0 :
        neg += 1
        phi_j_y_0 [ 0 ] [ int ( train_data [ i ] [ 0 ] / 100 ) ] += 1
        phi_j_y_0 [ 1 ] [ int ( train_data [ i ] [ 1 ] / 100 ) ] += 1
        phi_j_y_0 [ 2 ] [ int ( train_data [ i ] [ 2 ] / 100 ) ] += 1
    else :
        pos += 1
        phi_j_y_1 [ 0 ] [ int ( train_data [ i ] [ 0 ] / 100 ) ] += 1
        phi_j_y_1 [ 1 ] [ int ( train_data [ i ] [ 1 ] / 100 ) ] += 1
        phi_j_y_1 [ 2 ] [ int ( train_data [ i ] [ 2 ] / 100 ) ] += 1

phi_j_y_0 /= neg
phi_j_y_1 /= pos
phi_y_0 = (neg - 9) / (neg + pos - 18)
phi_y_1 = (pos - 9) / (pos + neg - 18)

correct = 0

for i in range ( len ( test_label ) ) :
    phi_x_y_1 = phi_j_y_1 [ 0 ] [ int ( test_data [ i ] [ 0 ] / 100 ) ] * phi_j_y_1 [ 1 ] [ int ( test_data [ i ] [ 1 ] / 100 ) ] * \
              phi_j_y_1 [ 2 ] [ int ( test_data [ i ] [ 2 ] / 100 ) ] * phi_y_1
    phi_x_y_0 = phi_j_y_0 [ 0 ] [ int ( test_data [ i ] [ 0 ] / 100 ) ] * phi_j_y_0 [ 1 ] [ int ( test_data [ i ] [ 1 ] / 100 ) ] * \
              phi_j_y_0 [ 2 ] [ int ( test_data [ i ] [ 2 ] / 100 ) ] * phi_y_0

    p_0 = phi_x_y_0 / ( phi_x_y_0 + phi_x_y_1 )
    p_1 = phi_x_y_1 / ( phi_x_y_0 + phi_x_y_1 )

    correct += ( ( p_1 >= p_0 ) == test_label [ i ] )

print ( "Correction: %.2f" % ( correct / len ( test_label ) * 100 ) + "%" )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )