# @Time: 6/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: feature_selection.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import seaborn
import matplotlib.pyplot as plt
from Tools.MI_Score import make_mi_scores

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

print ( make_mi_scores ( train_pca , train_label , False ) )

train_pca [ "Survived" ] = train.Survived

seaborn.relplot ( x = 0 , y = 10 , hue = "Survived" , data = train_pca )
plt.show ()