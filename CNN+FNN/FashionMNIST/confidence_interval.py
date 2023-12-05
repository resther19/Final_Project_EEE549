

#------------------ finding confidence interval by initializing different weights each time the model is instantiated----------------------


import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from CNN2layer import CNNmodel
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.init as init
from sklearn.model_selection import train_test_split


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.ones_(m.weight)
        init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        init.zeros_(m.bias)

def CNNmodel1():
    model = nn.Sequential(
        # First convolution layer
        nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),

        nn.MaxPool2d(kernel_size=2, stride=2),

        # Flatten
        nn.Flatten(),

        # Fully connected layers
        nn.Linear(128 * 13 * 13, 128),  

        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.15),

        nn.Linear(128, 10)  
    )

    model.apply(initialize_weights)
    return model


import torch.nn as nn
import torch.nn.init as init

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.ones_(m.weight)
        init.zeros_(m.bias)

def CNNmodel2():
    model = nn.Sequential(
        # First convolution layer
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(0.3),

        # Second convolution layer
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(0.3),

        # Flatten
        nn.Flatten(),

        # Fully connected layers
        nn.Linear(128 * 4 * 4, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),

        nn.Linear(128, 10)
    )

    # Apply the weight initialization
    model.apply(initialize_weights)
    
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=data_transforms)
train_data = train_dataset.data
train_data = train_data.unsqueeze(1).float()
train_targets = train_dataset.targets

train_data, valid_data, train_targets, valid_targets = train_test_split(
    train_data, train_targets, test_size=0.2, random_state=42)

train = torch.utils.data.TensorDataset(train_data, train_targets)
trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

valid = torch.utils.data.TensorDataset(valid_data, valid_targets)
validloader = torch.utils.data.DataLoader(valid, batch_size = 64, shuffle = False)

model = CNNmodel2()
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9 )
valid_accuracy_list = []
for testcase in range(100):
    print(testcase)
    model = CNNmodel1()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9 )
    epochs = 10
    for e in range(epochs):
        running_loss = 0
        correct = 0
        for images, labels in trainloader:
            images, labels = images.to(device),labels.to(device)
            optimizer.zero_grad()            
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()        
            running_loss += loss.item()
        
        #train_accuracy = 100. * correct / len(trainloader.dataset)
        #print(f"epoch {e + 1} : Train accuracy: {train_accuracy}")



    model.eval()
    true_labels = []
    predictions = []
    correct = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)

            log_ps = model(images)
            _, predicted = torch.max(log_ps, 1)
            correct = (predicted == labels).sum().item()
            accuracy += correct
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
        valid_accuracy = accuracy *100 / len(validloader.dataset)
        valid_accuracy_list.append(valid_accuracy)
        print(valid_accuracy)

print(valid_accuracy_list)
# %%
alpha = 0.95
lower_p = ((1.0 - alpha) / 2.0) * 100
upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100

lower, upper = np.percentile(valid_accuracy_list, [lower_p, upper_p])
print(f"95% Confidence interval for the validation accuracy: [{lower:.2f}, {upper:.2f}]")
