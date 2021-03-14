# @Time: 2/21/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Prediction.py

import pandas as pd
import tensorflow as tf
import numpy as np

train = pd.read_csv ( "train.csv" )
test = pd.read_csv ( "test.csv" )

train_data = train [ [ "PassengerId" , "Pclass" , "Name" , "Sex" , "Age" , "SibSp" , "Parch" , "Ticket" , "Fare" , "Cabin" , "Embarked" ] ].values
train_label = train [ "Survived" ].values
test_data = test [ [ "PassengerId" , "Pclass" , "Name" , "Sex" , "Age" , "SibSp" , "Parch" , "Ticket" , "Fare" , "Cabin" , "Embarked" ] ].values

model = tf.keras.Sequential ( [
    tf.keras.layers.Dense ( 3 , activation = 'relu' ) ,
    tf.keras.layers.Dense ( 2 , activation = "softmax" )
] )

model.compile ( optimizer = "adam" , loss = "sparse_categorical_crossentropy" , metrics = [ "acc" ] )

model.fit ( train_data , train_label , epochs = 1000 , batch_size = 1 )

predict = model.predict ( test_data )

df = pd.DataFrame ( { "PassengerId" : test_data [ : ] [ 0 ] , "Survived" : predict } )
df.to_csv ( "prediction.csv" , index = False )