# %%
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import torch
from kfold import kfoldtrain
from data_processing import adult
import matplotlib.pyplot as plt


# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

X_train, y_train, X_test, y_test = adult()
# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.LongTensor(y_train.values)




avg_acc = kfoldtrain(           #this function we used to tune the model with self created kfold function
                                #we would change the model in the models.py and change the optimizer in kfold.py
        X_train_tensor, 
        y_train_tensor, 
        device, 
        learning_rate = 0.001,
        epochs = 50,
        momentum=0.9
        )

print(avg_acc)

