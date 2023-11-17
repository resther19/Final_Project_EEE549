# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:21:33 2023

@author: naima
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder
# from ucimlrepo import fetch_ucirepo 

#%% load UCI Adult dataset from downloaded zip folder
labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
          'marital-status', 'occupation', 'relationship', 'race', 'sex', 
          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 
          'income']

train_df = pd.read_csv('adult.data', na_values='?', header=None, names=labels, skipinitialspace=True)
test_df = pd.read_csv('adult.test', na_values='?', header=None, names=labels, skipinitialspace=True)
train_df.dropna(axis=0, how='any', inplace=True)
test_df.dropna(axis=0, how='any', inplace=True)

# keep a copy to compare with processed data
data_copy_2 = pd.concat([train_df, test_df], axis=0)

#%% analyse features
target = train_df.values[:,14] # change column index 0-14
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))

#train_df.head()
#test_df.head()

#%% drop redundant column
train_df.drop(labels=['education'], axis=1, inplace=True)
test_df.drop(labels=['education'], axis=1, inplace=True)

#%% convert 'income' to integer
salary_map={'<=50K':0,'>50K':1}
salary_map2={'<=50K.':0,'>50K.':1}

train_df['income']=train_df['income'].map(salary_map).astype(int)
test_df['income']=test_df['income'].map(salary_map2).astype(int)

#%% process 'capital-gain' and 'capital-loss'
train_df.loc[(train_df['capital-gain'] > 0),'capital-gain'] = 1    #capital loss and gain has a lot of zero values 
train_df.loc[(train_df['capital-gain'] == 0 ,'capital-gain')]= 0

test_df.loc[(test_df['capital-gain'] > 0),'capital-gain'] = 1    #capital loss and gain has a lot of zero values 
test_df.loc[(test_df['capital-gain'] == 0 ,'capital-gain')]= 0

train_df.loc[(train_df['capital-loss'] > 0),'capital-loss'] = 1
train_df.loc[(train_df['capital-loss'] == 0 ,'capital-loss')]= 0

test_df.loc[(test_df['capital-loss'] > 0),'capital-loss'] = 1
test_df.loc[(test_df['capital-loss'] == 0 ,'capital-loss')]= 0

#%% convert object type data to integers
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        encoder = LabelEncoder()
        encoder.fit(train_df[col])
        train_df[col] = encoder.transform(train_df[col])
        encoder.fit(test_df[col])
        test_df[col] = encoder.transform(test_df[col])

#%% store processed data
# train_df.to_csv('adult-train.csv', index=False)
# test_df.to_csv('adult-test.csv', index=False)

#%% separate X and y
X_train = train_df.drop(['income'], axis=1).to_numpy()
y_train = train_df['income'].to_numpy()
X_test = test_df.drop(['income'], axis=1).to_numpy()
y_test = test_df['income'].to_numpy()

print(X_train.shape)
print(y_train.shape)



#%% model: logistic regression
from sklearn.linear_model import LogisticRegression

"""
class sklearn.linear_model.LogisticRegression(penalty='l2', 
                                              C=1.0,
                                              solver='lbfgs', 
                                              max_iter=100)
"""

model = LogisticRegression(penalty='l2', C=0.00001, solver='newton-cholesky', max_iter=1000)
model.fit(X_train, y_train)
model.score(X_train, y_train)

#%% tune various hyperparameters using GridSearchCV()
from sklearn.model_selection import GridSearchCV

# define the parameter grid
param_grid = {
    'penalty': ['l1', 'l2'],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'max_iter': [100, 500, 1000]
}

# grid search for best hyperparameters with 5-fold cross-validation
model = LogisticRegression()
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

#%%
# print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

#%%
# compute the accuracy score for each hyperparameter combination
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print(f"Accuracy: {mean:.3f} (±{std*2:.3f}) for {params}")

#%% K-fold for optimal C using the best hyperparameters given by grid search
from sklearn.model_selection import KFold

# define the values of C
C_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]

# define fixed hyperparameters
params = grid_search.best_params_

# number of folds
num_folds = 5

# initialize KFold
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# store average accuracy for each C
avg_accuracies = []

for C in C_values:
    accuracies = []

    # Loop over folds
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Create and train the model
        model = LogisticRegression(penalty=params['penalty'], C=C, 
                                   solver=params['solver'], 
                                   max_iter=params['max_iter'])
        model.fit(X_train_fold, y_train_fold)

        # Calculate accuracy on the test fold
        accuracy = model.score(X_val_fold, y_val_fold)
        accuracies.append(accuracy)

    # Calculate and print the average accuracy for the current C value
    avg_accuracy = np.mean(accuracies)
    print(f"C={C}: Average Accuracy = {avg_accuracy:.3f}")
    avg_accuracies.append(avg_accuracy)
    
#%%
print(avg_accuracies)

"""
avg_accuracies = [0.7554540434330896, 0.7833367627474179, 0.8026324874991262, 
                  0.8129102766350063, 0.8119487986278827, 0.8119487546610353, 
                  0.8117498376524166]
"""

#%%
size = 'x-large'
fig, ax = plt.subplots(figsize=(9,6))

ax.set_title('Log.Reg. (Adult) Accuracy vs. C', fontsize=size)
ax.semilogx(C_values, avg_accuracies, label='Avg. Validation Accuracy')
ax.legend()
ax.set_xlabel('C', fontsize=size)
ax.set_ylabel('Accuracy', fontsize=size)
ax.tick_params(axis='both', labelsize=size)
ax.grid(axis='both')

plt.savefig('LogReg_Adult.pdf', bbox_inches='tight')

#%% final evaluation with best hyperparameters
C_opt = 0.01

model = LogisticRegression(penalty=params['penalty'], C=C_opt, 
                           solver=params['solver'], 
                           max_iter=params['max_iter'])
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

"""
train_accuracy = 0.8134739075658113
test_accuracy = 0.8124169986719787
"""

#%% 
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix

# make predictions
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print or use the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

#%%
from sklearn.metrics import roc_curve, auc

# probability estimates for positive class
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# plt.savefig('LogReg_Adult_ROC-AUC.pdf', bbox_inches='tight')

#%%
import seaborn as sns

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# plt.savefig('LogReg_Adult_ConfMat.pdf', bbox_inches='tight')