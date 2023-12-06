# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, LearningCurveDisplay, ValidationCurveDisplay, ShuffleSplit
from sklearn.preprocessing import label_binarize

# Define transformations
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.numpy().flatten())])

# Load the Fashion MNIST dataset
trainset = datasets.FashionMNIST(root='~/.pytorch/F_MNIST_data/', train=True, download=True, transform=transform)
testset = datasets.FashionMNIST(root='~/.pytorch/F_MNIST_data/', train=False, download=True, transform=transform)

# Prepare the data
X_train, y_train = zip(*[(image, label) for image, label in trainset])
X_test, y_test = zip(*[(image, label) for image, label in testset])

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Flatten the images
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Parameter grid
param_grid = {
    'C': [11, 13, 15],
    'gamma': ['auto', 0.1] 
}

# Grid search for hyperparameter optimization
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train_pca, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
best_C = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']
best_accuracy = grid_search.best_score_
print(f"Accuracy with best C: {best_accuracy:.4f}")

# Extracting validation scores for the heatmap
scores_matrix = grid_search.cv_results_['mean_test_score'].reshape(len(param_grid['C']), len(param_grid['gamma']))

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(scores_matrix, annot=True, xticklabels=param_grid['gamma'], yticklabels=param_grid['C'], cmap='viridis')
plt.title('Validation Accuracy for each C and Gamma combination')
plt.xlabel('Gamma')
plt.ylabel('C')
plt.show()

# Predictions
y_pred = grid_search.predict(X_test_pca)

# Classification report
print(classification_report(y_test, y_pred, digits=3))

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve for each class
n_classes = len(np.unique(y_test))
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
y_pred_bin = label_binarize(y_pred, classes=np.arange(n_classes))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at these points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average and compute AUC
mean_tpr /= n_classes

from itertools import cycle


# Plot all ROC curves
plt.figure(figsize=(10, 8))
colors = cycle(['orange', 'red', 'green', 'cyan', 'olive', 'yellow', 'black', 'blue', 'magenta', 'indigo'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc="lower right")
plt.show()

# %%
# Define the best SVM model using the found hyperparameters
best_svm = SVC(kernel='rbf', C=best_C, gamma=best_gamma)

# Define learning curve parameters
train_sizes = np.linspace(0.1, 1.0, 5)

# Plot the learning curve
fig, ax = plt.subplots(figsize=(10, 6))

LearningCurveDisplay.from_estimator(
    estimator=best_svm,
    X=X_train_pca,
    y=y_train,
    train_sizes=train_sizes,
    n_jobs=4,
    ax=ax,
    scoring='accuracy'
)

plt.title(f"Learning Curve for {best_svm.__class__.__name__} with PCA")
plt.show()



# %%
ValidationCurveDisplay.from_estimator(
   estimator=SVC(kernel='rbf'), X=X_train_pca, y=y_train, param_name="C", param_range=np.logspace(-3, 3, 3), scoring='accuracy'
)
# %%