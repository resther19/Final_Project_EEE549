# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:05:46 2023

@author: naima
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from ucimlrepo import fetch_ucirepo 

#%% load Wisconsin Breast Cancer dataset using ucimlrepo
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets

print(X.dtypes)
print(y.dtypes)

#%% analyse features
target = y['Diagnosis']
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))
    
#%% convert categorical labesl to integer
label_map={'M':1, 'B':0}
y['Diagnosis']=y['Diagnosis'].map(label_map).astype(int)

#%% split train and test sets
X = X.to_numpy()
y = y.to_numpy()
y = np.squeeze(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#%% model: logistic regression
model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)
model.score(X_train, y_train)

"""
accuracy = 0.955078125
"""

#%% standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)
model.score(X_train, y_train)

"""
accuracy = 0.98828125
"""



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
                                   solver='liblinear', 
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
avg_accuracies = [0.9335808109651629, 0.9335808109651629, 0.9433276223110603, 
                  0.9667237768893966, 0.9804302303445651, 0.9745859508852085, 
                  0.964820102798401]
"""

#%%
size = 'x-large'
fig, ax = plt.subplots(figsize=(9,6))

ax.set_title('Log.Reg. (WDBC) Accuracy vs. C', fontsize=size)
ax.semilogx(C_values, avg_accuracies, label='Avg. Validation Accuracy')
ax.legend()
ax.set_xlabel('C', fontsize=size)
ax.set_ylabel('Accuracy', fontsize=size)
ax.tick_params(axis='both', labelsize=size)
ax.grid(axis='both')

# plt.savefig('LogReg_WDBC.pdf', bbox_inches='tight')

#%% final evaluation with best hyperparameters
C_opt = 0.1

model = LogisticRegression(penalty=params['penalty'], C=C_opt, 
                           solver='liblinear', 
                           max_iter=params['max_iter'])
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

"""
train_accuracy = 0.984375
test_accuracy = 0.9824561403508771
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