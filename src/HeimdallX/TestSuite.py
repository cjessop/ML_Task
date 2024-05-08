# HEIMDALL

print("HEIMDALL \n")


from ML_packaging import ML_meta
from ML_packaging import BasePredictor
from ML_packaging import ML, ML_post_process

from BaseMLClasses import PipelineLoader
from pipeline import pipeline

import yaml
import numpy as np
import pandas as pd
import os
import sys
import unittest
import warnings

df = pd.read_csv("heart.csv")
target_col = ["output"]
#print("The categorial cols are : ", cat_cols)
#print("The continuous cols are : ", con_cols)
#print("The target variable is :  ", target_col)

with open("runtime.yml", 'r') as f:
    data_in = yaml.load(f, Loader=yaml.SafeLoader)
    data = data_in['-data']['data']
    df = pd.read_csv(data)
    encode = data_in['-data']['encode']

    model = data_in['-model']['model']
    all = data_in['-model']['all']
    search = data_in['-model']['search']
    save = data_in['-model']['save']
    save_name = data_in['-model']['save_model_name']
    cross_val = data_in['-model']['cross_val']
    cm = data_in['-model']['confusion']

    saved_model = data_in['-post_processing']['saved_model']
    predict = data_in['-post_processing']['predict']
    target = data_in['-target']['target']
    cat_cols = data_in['-post_processing']['cat_cols']
    con_cols = data_in['-post_processing']['con_cols']
    feature = data_in['-post_processing']['feature']
    print(target)


if data_in['-architecture']['architecture'].lower() == 'keras':
    # try:
    ML = ML_meta(df, all=all, model=False, CNN=True, on_GPU=True)
    # except RuntimeError:
    #     raise RuntimeError
    ML.apply_CNN(CNN_flag=True)

ML = ML_meta(df, all=all, model=model, target=target, search=search, cross_val=cross_val)
ML.apply_single_model(save_model=save, save_model_name=save_name, cm=cm)



post_process = ML_post_process(data=df, saved_model=saved_model, predict=predict)
# Pipeline = pipeline.read("runtime.yml")
# print(Pipeline)

# assert loader.load_pipeline_yaml("runtime.yml") == {}

# Splitting the data into train and test
#ML = ML_meta(df, all=False, model='EC', target='output', cross_val=True, search=None)
#ML.apply_all_models(flag=True)
#ML.apply_single_model(save_model=True, save_model_name='gbc_model.pkl', cm=True)

#post_process = ML_post_process(data=df, saved_model=None, predict=False, target='output', con_cols=con_cols, feature='caa')
# post_process.data_info()

#post_process.univariate_analysis(output_plot='kde')