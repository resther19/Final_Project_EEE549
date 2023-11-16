# %%
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
from data_processing import cancer

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

X_train, X_test, y_train, y_test = cancer()

#------------------------hyperparameter tuning------------------------------- 

#tuning learning rate
learning_rate = [0.0001, 0.001, 0.005, 0.003, 0.007, 0.01, 0.02, .05]
avg_acc_LR = np.zeros(len(learning_rate))
for k in range(len(learning_rate)):
    print('for learning rate of ' , learning_rate[k])
    avg_acc_LR[k] = kfoldtrain(
        X_train, 
        y_train, 
        device, 
        learning_rate[k],
        epochs = 100,
        momentum=0.9
        )

#tuning epochs
epoch_list = [50, 100, 150, 200, 300]
avg_acc_Epochs = np.zeros(len(epoch_list))
for k in range(len(epoch_list)):
    print('for epoch numbers ' , epoch_list[k])
    avg_acc_Epochs[k] = kfoldtrain(
        X_train, 
        y_train,  
        device, 
        learning_rate = 0.001,
        epochs = epoch_list[k],
        momentum = 0.9
        )
# %%
#tuning momentum 
momentum_list = [ 0.1, 0.3, 0.5, 0.7, 0.9]
avg_acc_momentum = np.zeros(len(momentum_list))
for k in range(len(momentum_list)):
    print('for momentum' , momentum_list[k])
    avg_acc_momentum[k] = kfoldtrain(
        X_train, 
        y_train,  
        device,
       learning_rate = 0.001,
        epochs = 100,
        momentum = momentum_list[k]
        )


plt.figure(figsize=(10, 6))
plt.title('K-Fold Validation accuracy Vs Learning rate')
plt.xlabel('Learning Rate')
plt.ylabel('Avg Cross-validation Accuracy')
plt.plot(learning_rate, avg_acc_LR, marker = 'o' , color='blue')
plt.savefig("LRtuning.png", dpi=300)
plt.show()



plt.figure(figsize=(10, 6))
plt.title('K-Fold Validation accuracy Vs epochs')
plt.xlabel('epochs')
plt.ylabel('Avg Cross-validation Accuracy')
plt.plot(epoch_list, avg_acc_Epochs, marker = 'x' , color='red')
plt.savefig("epochtraining.png", dpi=300)
plt.show()
# %%
plt.figure(figsize=(10, 6))
plt.title('K-Fold Validation accuracy Vs momentum')
plt.xlabel('momentum')
plt.ylabel('Avg Cross-validation Accuracy')
plt.plot(momentum_list, avg_acc_momentum, marker = '*' , color='black')
plt.savefig("momentumtraining.png", dpi=300)
plt.show()



#######    ultimate model with all the best parameters 

#------> goes here 
# %%
