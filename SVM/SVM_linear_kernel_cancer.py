# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve
from sklearn.model_selection import train_test_split, LearningCurveDisplay, ValidationCurveDisplay, GridSearchCV


# Load and preprocess the Breast Cancer dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
cancer_df = pd.read_csv(url, header=None, names=column_names)

# Encode labels: M (Malignant) -> 1, B (Benign) -> 0
label_encoder = LabelEncoder()
cancer_df['Diagnosis'] = label_encoder.fit_transform(cancer_df['Diagnosis'])

# Split into features and labels
X = cancer_df.iloc[:, 2:].values
y = cancer_df['Diagnosis'].values

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5 fold validation
param_range = np.logspace(-6, 3, 10)
param_grid = {'C': param_range}
grid_search = GridSearchCV(SVC(kernel="linear"), param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

best_C = grid_search.best_params_['C']
best_accuracy = grid_search.best_score_

print(f"Best parameter C: {best_C}")
print(f"Accuracy with best C: {best_accuracy:.4f}")

# Train final model with the best C value and probability estimates
final_model = SVC(kernel='linear', C=best_C, probability=True)
final_model.fit(X_train, y_train)  

# Evaluate the final model on the test set
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]  
print("\nTest Set Evaluation")
print(classification_report(y_test, y_pred, digits=3))

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True)
plt.title('Confusion Matrix for SVM with Linear Kernel')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('LinealKernel_cm_cancer.pdf')

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.savefig('LinealKernel_ROC_cancer.pdf')

ValidationCurveDisplay.from_estimator(
   estimator=SVC(kernel='linear'), X=X_train, y=y_train, param_name="C", param_range=np.logspace(-6, 3, 10), scoring='accuracy'
)
plt.title('5-fold validation for the hyperparameter C')
plt.savefig('LinealKernel_C_parameter_cancer.pdf')

n_iterations = 300
bootstrap_results = []
param_range = np.logspace(-6, 3, 10)
param_grid = {'C': param_range}

# Bootstrap process
for i in range(n_iterations):
    # Resample the training data with replacement
    indices = np.random.choice(len(X_train), len(X_train))
    X_train_resampled = X_train[indices]
    y_train_resampled = y_train[indices]

    
    grid_search = GridSearchCV(SVC(kernel="linear"), param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train_resampled, y_train_resampled)
    avg_accuracy = grid_search.cv_results_['mean_test_score']
    C = grid_search.best_params_
    bootstrap_results.append((C, avg_accuracy))


# Calculate the confidence interval
mean_accuracies = [np.max(scores) for _, scores in bootstrap_results]

lower_bound = np.percentile(mean_accuracies, 2.5)
upper_bound = np.percentile(mean_accuracies, 97.5)

print(f"95% confidence interval for the mean accuracy score: [{lower_bound:.4f}, {upper_bound:.4f}]")

# Define the best SVM model using the found hyperparameters
best_svm = SVC(kernel='linear', C=best_C)

# Define learning curve parameters
train_sizes = np.linspace(0.1, 1.0, 10)

# Plot the learning curve
fig, ax = plt.subplots(figsize=(10, 6))

LearningCurveDisplay.from_estimator(
    estimator=best_svm,
    X=X_train,
    y=y_train,
    train_sizes=train_sizes,
    n_jobs=4,
    ax=ax,
    scoring='accuracy'
)

plt.title(f"Learning curve for SVM with linear kernel")
plt.savefig('LinealKernel_learning_cancer.pdf')

# %%
