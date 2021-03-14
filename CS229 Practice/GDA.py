# A simple Gaussian Discriminant Analysis model for classification
# @Time: 3/12/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: GDA.py

import numpy as np
import time
import pandas as pd

train = pd.read_csv ( "classification_train.csv" )
test = pd.read_csv ( "classification_test.csv" )
# train [ "c" ] = np.ones ( len ( train ) )
# test [ "c" ] = np.ones ( len ( test ) )

train_data = train [ [ "x1" , "x2" , "x3" ] ].values
train_label = train [ "y" ].values
test_data = test [ [ "x1" , "x2" , "x3" ] ].values
test_label = test [ "y" ].values

init = time.time ()

phi = 0.0

num_pos = 0.0
num_neg = 0.0
mu0 = np.zeros ( len ( train_data [ 0 ] ) )
mu1 = np.zeros ( len ( train_data [ 0 ] ) )

for i in range ( len ( train_label ) ) :
    phi += (train_label [ i ] == 1)

    num_pos += (train_label [ i ] == 1)
    num_neg += (train_label [ i ] == 0)
    mu0 += ( train_label [ i ] == 0 ) * train_data [ i ]
    mu1 += ( train_label [ i ] == 1 ) * train_data [ i ]

phi /= len ( train_label )
mu0 /= num_neg
mu1 /= num_pos

sigma = np.zeros ( ( len ( train_data [ 0 ] ) , len ( train_data [ 0 ] ) ) )

for i in range ( len ( train_label ) ) :
    if train_label [ i ] == 0 :
        sigma += np.dot ( np.reshape ( train_data [ i ] - mu0 , ( len ( train_data [ i ] ) , 1 ) ) , np.reshape ( train_data [ i ] - mu0 , ( 1 , len ( train_data [ i ] ) ) ) )
    else :
        sigma += np.dot ( np.reshape ( train_data [ i ] - mu1 , ( len ( train_data [ i ] ) , 1 ) ) , np.reshape ( train_data [ i ] - mu1 , ( 1 , len ( train_data [ i ] ) ) ) )

correct = 0

for i in range ( len ( test_label ) ) :
    p_x_y_0 = 1 / ((2 * np.pi) ** len ( train_data [ 0 ] ) * np.sqrt ( np.linalg.det ( sigma ) )) * np.exp (
        -.5 * np.dot ( np.dot ( np.transpose ( test_data [ i ] - mu0 ) , np.linalg.inv ( sigma ) ) , (test_data [ i ] - mu0) ) )
    p_x_y_1 = 1 / ((2 * np.pi) ** len ( train_data [ 0 ] ) * np.sqrt ( np.linalg.det ( sigma ) )) * np.exp (
        -.5 * np.dot ( np.dot ( np.transpose ( test_data [ i ] - mu1 ) , np.linalg.inv ( sigma ) ) , (test_data [ i ] - mu1) ) )

    p_0 = p_x_y_0 * ( 1 - phi ) / ( p_x_y_0 + p_x_y_1 )
    p_1 = p_x_y_1 * phi / ( p_x_y_0 + p_x_y_1 )

    if ( p_0 <= p_1 ) == test_label [ i ] :
        correct += 1

print ( "Correction: %.2f" % ( correct / len ( test_label ) * 100 ) + "%" )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )