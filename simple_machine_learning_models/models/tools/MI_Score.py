# @Time: 6/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: MI_Score.py

from sklearn.feature_selection import mutual_info_regression
import pandas as pd

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features = discrete_features)
    mi_scores = pd.Series(mi_scores, name = "MI Scores", index = X.columns)
    mi_scores = mi_scores.sort_values(ascending = False)
    return mi_scores
