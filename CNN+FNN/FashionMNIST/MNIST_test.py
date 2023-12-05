# %%
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Define a transform to normalize the data

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),                 #Data augmentation 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=data_transforms)
train_data = train_dataset.data
train_data = train_data.unsqueeze(1).float()      #introducing a channel dimension for CNN model fitting
train_targets = train_dataset.targets

transform_test = transforms.Compose([transforms.ToTensor(),
                                                               #no augmentation in the test data, not allowed.
                                transforms.Normalize((0.5,), (0.5,))])

test_dataset = datasets.FashionMNIST('.data/', download=True, train=False, transform=transform_test)
test_data = test_dataset.data # Add channel dimension if needed
test_data = test_data.unsqueeze(1).float()  # Add channel dimension if needed
test_targets = test_dataset.targets

train = torch.utils.data.TensorDataset(train_data, train_targets)
trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

test = torch.utils.data.TensorDataset(test_data, test_targets)
testloader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

# Instantiate the model
model = CNNmodel()
model = model.to(device)
# Define the loss

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9 )

# Train the network
epochs = 50
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
    
    train_accuracy = 100. * correct / len(trainloader.dataset)

    print(f"epoch {e + 1} : Train accuracy: {train_accuracy}")


# Testing the model

model.eval()
true_labels = []
predictions = []
correct = 0
accuracy = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        log_ps = model(images)
        _, predicted = torch.max(log_ps, 1)
        correct = (predicted == labels).sum().item()
        accuracy += correct
        true_labels.extend(labels.cpu().numpy())
        predictions.extend(predicted.cpu().numpy())

print(f'Test Accuracy: {accuracy *100 / len(testloader.dataset)}')

conf_mtx = confusion_matrix(true_labels, predictions)
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predictions, average=None)



# %%

plt.figure(figsize=(10, 8))
sns.heatmap(conf_mtx, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig("confusionMatrix_1layer.png", dpi=300)
plt.show()

for i, (p, r, f) in enumerate(zip(precision, recall, f1_score)):
    print(f"Class {i}: Precision: {p:.4f}, Recall: {r:.4f}, F1-score: {f:.4f}")
