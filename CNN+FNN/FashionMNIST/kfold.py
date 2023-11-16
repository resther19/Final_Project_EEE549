# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from CNN2layer import CNNmodel

def kfoldtrain(train_data, train_targets, device, learning_rate, epochs, momentum):
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=False)

    kfold_val_losses = []
    kfold_val_accuracies = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_data)):
        print(f"FOLD {fold}")
        print("------------------------------")

        # Subset data
        train_subdata = train_data[train_ids].to(device)
        train_subtargets = train_targets[train_ids].to(device)
        val_subdata = train_data[val_ids].to(device)
        val_subtargets = train_targets[val_ids].to(device)

        # Initialize dataloaders
        train = torch.utils.data.TensorDataset(train_subdata, train_subtargets)
        val = torch.utils.data.TensorDataset(val_subdata, val_subtargets)
        trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
        valloader = torch.utils.data.DataLoader(val, batch_size=64, shuffle=False)

        # Initialize network and optimizer
        CNN = CNNmodel()
        model = CNN.to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        # Training loop
        for epoch in range(epochs):
            train_loss = 0
            correct = 0
            model.train()
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = nn.functional.cross_entropy(output, target)
                train_loss += loss
                loss.backward()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()     
                optimizer.step()
            train_loss /= len(trainloader.dataset)
            train_accuracy = 100. * correct / len(trainloader.dataset)

            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in valloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)

                    correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss /= len(valloader.dataset)
            val_accuracy = 100. * correct / len(valloader.dataset)
            #print(f"Epoch {epoch}: train Loss: {train_loss:.4f}, train Accuracy: {train_accuracy:.2f}%")
            print(f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        kfold_val_losses.append(val_loss)
        kfold_val_accuracies.append(val_accuracy)
    print("K-Fold Cross-Validation Complete!")
    avg_accuracy = np.mean(kfold_val_accuracies)




    return avg_accuracy
# %%
