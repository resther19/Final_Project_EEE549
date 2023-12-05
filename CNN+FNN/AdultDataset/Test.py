# %%
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from data_processing import adult
from models import FNN1
from models import FNN2
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

X_train, y_train, X_test, y_test = adult()

X_train = torch.FloatTensor(X_train.values)
y_train = torch.LongTensor(y_train.values)

X_test = torch.FloatTensor(X_test.values)
y_test = torch.LongTensor(y_test.values)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

trainloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

#**************************training loop********************************
net = FNN2().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs = 200
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    true_labels = []
    predictions = []
    correct = 0
    total = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Epoch {epoch+1}, acc: {correct/total}")

# ***********************Testing loop**********************************
net.eval()
true_labels = []
predictions = []
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        true_labels.extend(labels.cpu().numpy())
        predictions.extend(predicted.cpu().numpy())

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy}%")

precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predictions, average=None)

for i, (p, r, f) in enumerate(zip(precision, recall, f1_score)):
    print(f"Class {i}: Precision: {p:.4f}, Recall: {r:.4f}, F1-score: {f:.4f}")

conf_mtx = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mtx, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig("AdultConfusionMatLaye.png")
plt.show()
