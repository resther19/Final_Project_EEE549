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
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



class FNN2(nn.Module):
    def __init__(self, dropout_prob=0.2): 
        super(FNN2, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(14, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 2)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights here using your chosen method
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        # Define forward pass
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class FNN1(nn.Module):
    def __init__(self): 
        super(FNN1, self).__init__()
        # Define the layer
        self.fc1 = nn.Linear(14, 2)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights for fc1 layer
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x):
        # Define forward pass
        x = self.fc1(x)
        return x

    
    

X_train, y_train, X_test, y_test = adult()

X_train = torch.FloatTensor(X_train.values)
y_train = torch.LongTensor(y_train.values)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_val, y_val)

trainloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
validloader = DataLoader(valid_dataset, batch_size=100, shuffle=False)

valid_accuracy_list = []
for testcase in range(300):
    print(testcase)
    #**************************training loop********************************
    net = FNN2().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    epochs = 80
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

    # **********************validation loop**********************************
    net.eval()
    true_labels = []
    predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())


        valid_accuracy = 100 * correct / total
        valid_accuracy_list.append(valid_accuracy)
        print(valid_accuracy)

# %%
alpha = 0.95
lower_p = ((1.0 - alpha) / 2.0) * 100
upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100

lower, upper = np.percentile(valid_accuracy_list, [lower_p, upper_p])
print(f"95% Confidence interval for the validation accuracy: [{lower:.2f}, {upper:.2f}]")
