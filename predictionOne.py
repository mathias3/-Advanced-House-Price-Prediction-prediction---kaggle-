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




def rmse_cv(model, train, labels):
    rmse = np.sqrt(-cross_val_score(model, train, labels, scoring="neg_mean_squared_error", cv = 10))
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
