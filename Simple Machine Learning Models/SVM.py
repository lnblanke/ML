# A simple support vector machine for classification using scikit-learn model
# @Time: 3/21/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: SVM.py

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

data = make_classification ( n_samples = 1200 , n_classes = 2 , scale = 20 , n_clusters_per_class = 2 , n_features = 2 , n_redundant = 0 , class_sep = 1 )

train_data = data [ 0 ] [ : 1000 ]
train_label = data [ 1 ] [ : 1000 ]

test_data = data [ 0 ] [ 1000 : ]
test_label = data [ 1 ] [ 1000 : ]

if __name__ == '__main__':
    init = time.time ()

    svm = SVC ( kernel = "rbf" , C = 1 , degree = 3 , gamma = .001 )
    svm.fit ( train_data , train_label )

    correct = 0

    x0_1 = [] ; x0_2 = []
    x1_1 = [] ; x1_2 = []

    pred = svm.predict ( test_data )

    for i in range ( len ( test_label ) ) :
        if pred [ i ] == test_label [ i ] :
            correct += 1

        if pred [ i ] :
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

    plt.title ( "SVM" )

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