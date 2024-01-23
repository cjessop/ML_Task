# HORUS - High-Precision Object Classification and Radar Unidentified Signal Separation

print("HORUS \nMachine Learning and Artificial Intelligence classifier architecture \n")


from ML_packaging import ML_meta
from ML_packaging import BasePredictor
from ML_packaging import ML, ML_post_process

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
#print("The categorial cols are : ", cat_cols)
#print("The continuous cols are : ", con_cols)
#print("The target variable is :  ", target_col)

# Splitting the data into train and test
ML = ML_meta(df, all=True, model='GBC', target='output', cross_val=True, search='random')
#ML.apply_all_models(flag=True)
#ML.apply_single_model(save_model=True, save_model_name='gbc_model.pkl', cm=True)

post_process = ML_post_process(data=df, saved_model=None, predict=False, target='output', con_cols=con_cols, feature='caa')
# post_process.data_info()

post_process.univariate_analysis(output_plot='corr')