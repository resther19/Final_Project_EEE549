#******************** this module is used for hyperparameter tuning *************************************

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from kfold import kfoldtrain
import matplotlib.pyplot as plt
from data_processing import adult
from models import FNN2
from models import FNN1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
X_train, y_train, X_test, y_test = adult()
# Convert data to PyTorch tensors
X_train= torch.FloatTensor(X_train.values)
y_train = torch.LongTensor(y_train.values)


class PyTorchGridSearchModel(BaseEstimator):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size = 64):  #using basis values for super function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None  
        
    def fit(self, X, y):                             # defining my own fitting function for parameter optimization as I choose.
        # Here you should define your model architecture
        self.model = FNN1()  
        self.model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)    #we used Adam for this dataset because SGD was performing poorly
        
                                                            # Convert X and y to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
        return self

    def predict(self, X):                                    #inference taking place here
        self.model.eval()
        X_tensor =torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            output = self.model(X_tensor)
        return output.argmax(dim=1).cpu().numpy()

    def score(self, X, y):                                  #the metric used to find the optimum hyperparameter.
        # Here we'll use accuracy as the score
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# Defining the parameter grid
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.002, .005, .01],
    'epochs': [100, 150, 200],
    'batch_size': [50, 100, 150, 200, 250]
}

# Initialize GridSearching with cross validation 
grid_search = GridSearchCV(PyTorchGridSearchModel(), param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

with open('output.txt', 'w') as file:
    # Write the output to the file
    file.write("Best parameters found: {}\n".format(best_params))
    file.write("Best score found: {}\n".format(best_score))

#%%
#cv_results = grid_search.cv_results_
#for mean_score, params in zip(cv_results['mean_ test_score'], cv_results['params']):
#    print(params, '-> Mean CV Score:', mean_score)
