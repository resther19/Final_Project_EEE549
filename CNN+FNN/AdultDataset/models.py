
# %%
import torch.nn as nn
import torch
class FNN1(nn.Module):
    def __init__(self): 
        super(FNN1, self).__init__()

        self.fc1 = nn.Linear(14, 2)      # we can only use 14 here because of the feature size of this dataset limits this

    def forward(self, x):
        x = self.fc1(x)
        return x

class FNN2(nn.Module):
    def __init__(self, dropout_prob=0.2): 
        super(FNN2, self).__init__()

        self.fc1 = nn.Linear(14, 128)    # the 128 is chosen based on parameter tuning 

        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_prob)   # the dropout value was tuned using e iterative process

        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        return x

