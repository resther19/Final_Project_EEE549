# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns

class FNN1(nn.Module):
    def __init__(self): 
        super(FNN1, self).__init__()

        self.fc1 = nn.Linear(30, 2)

    def forward(self, x):
        x = self.fc1(x)
        return x

class FNN2(nn.Module):
    def __init__(self, dropout_prob=0.2): 
        super(FNN2, self).__init__()

        self.fc1 = nn.Linear(30, 120)

        self.bn1 = nn.BatchNorm1d(120)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(120, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        return x
      
def cancer():                #fetching validation data from this function cause we cannot touch the test data
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    y['Diagnosis'] = y['Diagnosis'].map({'M': 1, 'B': 0})
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)
    
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.LongTensor(y_train['Diagnosis'].values)
    X_val_tensor = torch.FloatTensor(X_val.values)
    y_val_tensor = torch.LongTensor(y_val['Diagnosis'].values)
    
    return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor


# Instantiate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, = cancer()

trainloader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=10, shuffle=True)
valloader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=10, shuffle=False)

from models import FNN1

# Training the model
net = FNN1().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs = 150

for epoch in range(epochs):
    net.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluating the model on the validation set
true_labels_val = []
predictions_val = []
net.eval()

with torch.no_grad():
    for inputs, labels in valloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        true_labels_val.extend(labels.cpu().numpy())
        predictions_val.extend(predicted.cpu().numpy())

# Bootstrapping for confidence interval calculation on validation data
accuracy_scores_val = []
n_bootstrap = 300

def calculate_accuracy(trues, preds):
    return (trues == preds).mean()

for _ in range(n_bootstrap):
    indices = np.random.randint(0, len(true_labels_val), len(true_labels_val))
    bootstrap_true = np.array(true_labels_val)[indices]
    bootstrap_pred = np.array(predictions_val)[indices]
    score = calculate_accuracy(bootstrap_true, bootstrap_pred)
    accuracy_scores_val.append(score)
    

alpha = 0.95
lower_p = ((1.0 - alpha) / 2.0) * 100
upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100

lower, upper = np.percentile(accuracy_scores_val, [lower_p, upper_p])
print(f"95% Confidence interval for the validation accuracy: [{lower:.2f}, {upper:.2f}]")
# %%
#print(np.max(accuracy_scores_val))
