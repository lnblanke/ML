# A practice sheet for sklearn APIs
# @Time: 4/2/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: sklearn-sheet.py

from matplotlib import pyplot as plt
import numpy as np

# Linear Regression
from sklearn.datasets import make_regression
from sklearn import linear_model

data = make_regression ( n_samples = 50 , n_features = 1 , bias = 1 , noise = 20 )

model = linear_model.LinearRegression ()

model.fit ( data [ 0 ] , data [ 1 ] )

plt.scatter ( data [ 0 ] , data [ 1 ] , c = "black" )
plt.xlabel ( 'X1' )
plt.ylabel ( 'X2' )

weight = model.coef_
bias = model.intercept_

x = np.arange ( -5 , 5 )
y = weight [ 0 ] * x + bias

plt.title ( "Linear Regression" )
plt.plot ( x , y )

plt.show ()