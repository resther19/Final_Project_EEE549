# %%
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
#from sklearn_pandas import DataFrameMapper
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

def adult():

    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'country', 'income']

    train_df = pd.read_csv("adult.data", header=None, names = columns, skipinitialspace=True, na_values= '?')
    test_df = pd.read_csv('adult.test', header=None, names=columns, skiprows= 1, skipinitialspace=True, na_values= '?') #first row has garbage values
    train_df.dropna(how='any',inplace=True)
    test_df.dropna(how='any',inplace=True)

    salary_map={'<=50K':0,'>50K':1}
    salary_map2={'<=50K.':0,'>50K.':1}

    train_df['income']=train_df['income'].map(salary_map).astype(int)     #mapping output columns to 1 and zero
    test_df['income']=test_df['income'].map(salary_map2).astype(int)    

    #test_df.head()
    #train_df.head()

       #df.income.unique()

    train_df.loc[(train_df['capital-gain'] > 0),'capital-gain'] = 1    #capital loss and gain has a lot of zero values 
    train_df.loc[(train_df['capital-gain'] == 0 ,'capital-gain')]= 0

    test_df.loc[(test_df['capital-gain'] > 0),'capital-gain'] = 1    
    test_df.loc[(test_df['capital-gain'] == 0 ,'capital-gain')]= 0

    train_df.loc[(train_df['capital-loss'] > 0),'capital-loss'] = 1
    train_df.loc[(train_df['capital-loss'] == 0 ,'capital-loss')]= 0
                                                                      #capital loss and gain has a lot of zero values 
    test_df.loc[(test_df['capital-loss'] > 0),'capital-loss'] = 1
    test_df.loc[(test_df['capital-loss'] == 0 ,'capital-loss')]= 0


    for col in train_df.columns:                    #changing categorical data to numerical data
        if train_df[col].dtype == 'object':
            encoder = LabelEncoder()
            train_df[col] = encoder.fit_transform(train_df[col])
            test_df[col] = encoder.transform(test_df[col]) 


    X_train= train_df.drop(['income'],axis=1)
    y_train = train_df['income']
    X_test= test_df.drop(['income'],axis=1)
    y_test = test_df['income']


    return X_train, y_train, X_test, y_test
# %%
