from ML_packaging import ML_meta
from ML_packaging import BasePredictor
from ML_packaging import ML

import numpy as np
import pandas as pd
import os
import sys
import unittest
import warnings

df = pd.read_csv("heart.csv")
cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
con_cols = ["age","trtbps","chol","thalachh","oldpeak"]
target_col = ["output"]
print("The categorial cols are : ", cat_cols)
print("The continuous cols are : ", con_cols)
print("The target variable is :  ", target_col)

# Splitting the data into train and test
ML = ML_meta(df, all=False, model='EC', target='output')
#ML.apply_all_models(flag=True)
ML.apply_single_model()