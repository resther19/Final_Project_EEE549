# %%
from sklearn.model_selection import StratifiedKFold
#from sklearn_pandas import DataFrameMapper
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from models import FNN1
from models import FNN2


def kfoldtrain( X_train_tensor, y_train_tensor, device, learning_rate, epochs, momentum):

    # Create a TensorDataset
    dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Number of splits and epochs
    n_splits = 5
    n_epochs = epochs

    # Lists to store fold-wise validation loss and accuracy
    val_losses = []
    val_accuracies = []

    # Initialize k-fold cross-validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # Start k-fold cross-validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train_tensor, y_train_tensor)):
        print(f"FOLD {fold}")
        print("--------------------------------")

        # Prepare dataloaders
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        trainloader = DataLoader(dataset, batch_size=10, sampler=train_subsampler)
        valloader = DataLoader(dataset, batch_size=10, sampler=val_subsampler)

        # Initialize network and optimizer
        net = FNN2()
        #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum = momentum)
        #optimizer = optim.Adam(net.parameters(), lr=learning_rate)#, momentum = momentum)
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)#, momentum = momentum)

        criterion = nn.CrossEntropyLoss()

        # Epoch loop
        valid_loss_list = []
        valid_acc_list = []
        for epoch in range(n_epochs):
            got = 0
            outputs_train = 0
            print(f"inside epoch: {epoch}")
            net.train()
            for batch in trainloader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs_train = net(inputs)
                loss = criterion(outputs_train, labels)
                loss.backward()
                optimizer.step()

            # Validation loop
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in valloader:
                    net.eval()
                    inputs, labels = batch
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()       

            val_loss /= len(valloader)
            accuracy = 100 * correct / total
            valid_loss_list.append(val_loss)
            valid_acc_list.append(accuracy)


            #print(f"Validation Loss: {val_loss}")
            print(f"Validation Accuracy: {accuracy}%")

        val_losses.append(min(valid_loss_list))
        val_accuracies.append(max(valid_acc_list))

    return np.mean(val_accuracies)
