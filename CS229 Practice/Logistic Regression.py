# A simple logistic regression model for classfication
# @Time: 3/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Logistic Regression.py

import numpy as np
import time
import pandas as pd

train = pd.read_csv ( "classification_train.csv" )
test = pd.read_csv ( "classification_test.csv" )
train [ "c" ] = np.ones ( len ( train ) )
test [ "c" ] = np.ones ( len ( test ) )

train_data = train [ [ "x1" , "x2" , "x3" , "c" ] ].values
train_label = train [ "y" ].values
test_data = test [ [ "x1" , "x2" , "x3" , "c" ] ].values
test_label = test [ "y" ].values

def sigmoid ( num ) :
    return 1.0 / ( 1 + np.exp ( -1 * num ) )

init = time.time()

cost = 1e7

rate = .5

weight = np.random.rand ( len ( train_data [ 0 ] ) )

count = 0

while True :
    gradient = 0

    for i in range ( len ( train_label ) ) :
        gradient += ( train_label [ i ] - sigmoid ( np.dot ( train_data [ i ] , weight ) ) ) * train_data [ i ]

    weight += rate * gradient

    correct = 0

    result = sigmoid ( np.dot ( train_data , weight ) )

    for i in range ( len ( train_label ) ) :
        if ( result [ i ] >= 0.5 ) == train_label [ i ] :
            correct += 1

    count += 1

    print ( "Epoch: %d Accuracy: %.5f" % (count , correct / len ( train_label ) ) )

    if correct / len ( train_label ) > .95 :
        break

correct = 0

result = sigmoid ( np.dot ( test_data , weight ) )

for i in range ( len ( test_label ) ) :
    if ( result [ i ] >= 0.5 ) == test_label [ i ] :
        correct += 1

print ( "Weight:" , weight )
print ( "Correction: %.2f" % ( correct / len ( test_label ) * 100 ) + "%" )
print ( "Time: %.5fms" % ( ( time.time() - init ) * 1000 ) )
