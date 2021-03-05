# @Time: 2/21/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Prediction.py

import csv
import tensorflow as tf
import numpy as np
import xlwt

train = csv.reader ( open ( "train.csv" ) )
test = csv.reader ( open ( "test.csv" ) )

train_data = np.zeros ( ( 891 , 7 ) )
train_label = np.zeros ( 891 )
test_data = np.zeros ( ( 419 , 7 ) )

i = 0

for line in train :
    if line [ 0 ] == "PassengerId" :
        continue
    train_label [ i ] = line [ 1 ]

    train_data [ i ] [ 0 ] = line [ 2 ]

    if line [ 4 ] == "male" :
        train_data [ i ] [ 1 ] = 0
    else :
        train_data [ i ] [ 1 ] = 1

    train_data [ i ] [ 2 ] = line [ 5 ]
    train_data [ i ] [ 3 ] = line [ 6 ]
    train_data [ i ] [ 4 ] = line [ 7 ]
    train_data [ i ] [ 5 ] = line [ 9 ]

    if line [ 11 ] == "S" :
        train_data [ i ] [ 6 ] = 0
    elif line [ 11 ] == "Q" :
        train_data [ i ] [ 6 ] = 1
    else :
        train_data [ i ] [ 6 ] = 2

    i += 1

print ( train_data [ 0 ] )
print ( train_data [ 1 ] )

i = 0

for line in test :
    if line [ 0 ] == "PassengerId" :
        continue

    test_data [ i ] [ 0 ] = line [ 1 ]

    if line [ 3 ] == "male" :
        test_data [ i ] [ 1 ] = 0
    else :
        test_data [ i ] [ 1 ] = 1

    test_data [ i ] [ 2 ] = line [ 5 ]
    test_data [ i ] [ 3 ] = line [ 6 ]
    test_data [ i ] [ 4 ] = line [ 7 ]
    test_data [ i ] [ 5 ] = line [ 9 ]

    if line [ 11 ] == "S" :
        test_data [ i ] [ 6 ] = 0
    elif line [ 11 ] == "Q" :
        test_data [ i ] [ 6 ] = 1
    else :
        test_data [ i ] [ 6 ] = 2

model = tf.keras.Sequential ( [
    tf.keras.layers.Dense ( 3 , activation = 'relu' ) ,
    tf.keras.layers.Dense ( 2 , activation = "softmax" )
] )

model.compile ( optimizer = "adam" , loss = "sparse_categorical_crossentropy" , metrics = [ "acc" ] )

model.fit ( train_data , train_label , epochs = 1000 , batch_size = 1 )

predict = model.predict ( test_data )

book = xlwt.Workbook ()
sheet = book.add_sheet ( "Sheet 1" )

sheet.write ( 0 , 0 , "PassengerId" )
sheet.write ( 0 , 1 , "Survived" )

for i in range ( 0 , len ( predict ) ) :
    sheet.write ( i + 1 , 0 , 892 + i )
    if predict [ i ] [ 0 ] > predict [ i ] [ 1 ] :
        sheet.write ( i + 1 , 1 , 1 )
    else :
        sheet.write ( i + 1 , 1 , 0 )

book.save ( "prediction.xls" )