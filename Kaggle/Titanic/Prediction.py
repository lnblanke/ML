# @Time: 2/21/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Prediction.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from sklearn.decomposition import PCA

train = pd.read_csv ( "train.csv" )
test = pd.read_csv ( "test.csv" )

drop_cols = [ "Name" , "PassengerId" , "Cabin" , "Ticket" ]

train = train.drop ( drop_cols , axis = 1 )
test = test.drop ( drop_cols , axis = 1 )

imputer = SimpleImputer ( missing_values = np.nan , strategy = "most_frequent" )
train_data = pd.DataFrame ( imputer.fit_transform ( train.drop ( "Survived" , axis = 1 ) ) )
test = pd.DataFrame ( imputer.transform ( test ) )

train_data.columns = train.drop ( "Survived" , axis = 1 ).columns
test.columns = train.drop ( "Survived" , axis = 1 ).columns

for col in train.select_dtypes ( "object" ) :
    train_data [ col ] , _ = train_data [ col ].factorize ()

for col in test.select_dtypes ( "object" ) :
    test [ col ] , _ = test [ col ].factorize ()

train_label = train.Survived

train_data = pd.DataFrame ( ColumnTransformer ( [ ( "one_hot_encoder" , OneHotEncoder () , [ "Pclass" , "Sex" , "Embarked" ] ) ] , remainder = "passthrough" ).fit_transform ( train_data ) )
test = pd.DataFrame ( ColumnTransformer ( [ ( "one_hot_encoder" , OneHotEncoder () , [ "Pclass" , "Sex" , "Embarked" ] ) ] , remainder = "passthrough" ).fit_transform ( test ) )

ss = StandardScaler ()
train_data = ss.fit_transform ( train_data )
test = ss.fit_transform ( test )

pca = PCA ( n_components = None )

train_pca = pd.DataFrame ( pca.fit_transform ( train_data ) )
test_pca = pd.DataFrame ( pca.transform ( test ) )

train_pca , valid_pca , train_label , valid_label = train_test_split ( train_pca , train_label , train_size = .8 )

model = tf.keras.Sequential ( [
    tf.keras.layers.Dense ( 8 , activation = "relu" ) ,
    tf.keras.layers.Dense ( 5 , activation = "relu" ) ,
    tf.keras.layers.Dense ( 1 , activation = "relu" )
] )

if __name__ == '__main__':
    model.compile ( optimizer = "adam" , loss = "binary_crossentropy" , metrics = [ "accuracy" ] )

    model.fit ( train_pca , train_label , epochs = 1000 , batch_size = 1 )
    pred = model.predict ( valid_pca )

    acc = 0

    for i in range ( 0 , len ( valid_label ) ) :
        acc += ( ( pred [ i ] >= .5 ) == valid_label.values [ i ] )

    print ( acc / len ( valid_label ) )

    pred = model.predict ( test_pca )

    output = pd.DataFrame ( { "PassengerId" : pd.read_csv ( "test.csv" ).PassengerId , "Survived" : np.argmax ( pred , axis = 1 ) } )
    output.to_csv ( "submission.csv" , index = False )