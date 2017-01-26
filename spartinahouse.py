# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:23:49 2017

@author: MO
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from scipy.stats import skew
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv as csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

def rmse_cv(model, train, labels):
    rmse = np.sqrt(-cross_val_score(model, train, labels, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

def selectFeatures(train, test):
    labels = train["SalePrice"]
    train.drop(["Id","SalePrice"], axis=1, inplace=True)
    test_ids = list(test["Id"])
    test.drop(["Id"], axis=1, inplace=True)
    # Select all numerical features
    num_features = train.dtypes[train.dtypes != 'object'].index
    train = train[num_features]
    test = test[num_features]
    return train, labels, test, test_ids

def dataCleaning(train, test, labels):
    labels = np.log1p(labels)
    skewed_feats = train.apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    train[skewed_feats] = np.log1p(train[skewed_feats])
    test[skewed_feats] = np.log1p(test[skewed_feats])
    # Fill nan for each feature
    train = train.fillna(train.mean())
    test = test.fillna(test.mean())
    return train, test, labels

df = pd.read_csv(open('../input/train.csv','rb'),header=0)
test_df = pd.read_csv(open('../input/test.csv','rb'),header=0)
train, labels, test, test_ids = selectFeatures(df, test_df)
train, test, labels = dataCleaning(train, test, labels)
train_data = train.values
test_data = test.values
labels_data = labels.values

# Train and Compare
print('Training...')
las = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(train_data, labels_data)
reg = RidgeCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(train_data, labels_data)
print(rmse_cv(las, train_data, labels_data).mean())
print(rmse_cv(reg, train_data, labels_data).mean())
dtrain = xgb.DMatrix(train_data, label=labels_data)
dtest = xgb.DMatrix(test_data)
params = {"max_depth":2, "eta":0.1}
#xgb_model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
#xgb_model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(train_data, labels_data)

print('Predicting...')
pred_reg = np.expm1(reg.predict(test_data).astype(float))
pred_xgb = np.expm1(model_xgb.predict(test_data).astype(float))
output = 0.5*pred_reg + 0.5*pred_xgb

sub = pd.DataFrame(output, columns=["SalePrice"])
sub.insert(0,'Id',test_ids)
sub.to_csv('prediction.csv',index=False)
print('Done.')