# %%
import numpy as np
import torch
from kfold import kfoldtrain
import matplotlib.pyplot as plt
from data_processing import cancer

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

X_train, X_test, y_train, y_test = cancer()

avg_acc = kfoldtrain(                 #this function we used to tune the model with self created kfold function
                                #we would change the model in the models.py and change the optimizer in kfold.py
        X_train, 
        y_train, 
        device, 
        learning_rate= 0.001,
        epochs = 50,
        momentum=0.9
        )

print(avg_acc)


# %%
