# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, ValidationCurveDisplay, LearningCurveDisplay, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import label_binarize

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
df = pd.read_csv(url, names=column_names, sep=',\s', na_values=["?"], engine='python')
df.dropna(inplace=True)

# Encoding categorical data
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column] = LabelEncoder().fit_transform(df[column])

# Splitting dataset into features and target variable
X = df.drop('income', axis=1).values
y = df['income'].values

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %%

# Parameter grid
param_grid = {
    'C': np.logspace(-4, 1, 6),
    'gamma': ['auto', 0.01, 0.1, 1]
}

# Grid search for hyperparameter optimization
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

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
sns.heatmap(scores_matrix,  fmt='.4f', annot=True, xticklabels=param_grid['gamma'], yticklabels=param_grid['C'], cmap='viridis')
plt.title('Validation accuracy for the hyperparameter grid')
plt.xlabel('Kernel Coefficient ($\gamma$)')
plt.ylabel('Regularization ($C$)')
plt.show()

# Training the final model with best C
final_model = SVC(kernel='rbf', C=best_C, gamma=best_gamma, probability=True)
final_model.fit(X_train, y_train)

# Evaluating the final model
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

print("\nTest Set Evaluation")
print(classification_report(y_test, y_pred, digits=3))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for SVM with RBF Kernel')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

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
plt.savefig('RBFKernel_ROC_adult.pdf')
# %%
# Define the best SVM model using the found hyperparameters
best_svm = SVC(kernel='rbf', C=best_C, gamma=best_gamma)

# Define learning curve parameters
train_sizes = np.linspace(0.1, 1.0, 5)

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

plt.title(f"Learning curve for SVM with RBF kernel")
plt.savefig('RBFKernel_learning_adult.pdf')

# %%
ValidationCurveDisplay.from_estimator(
   estimator=SVC(kernel='rbf'), X=X_train, y=y_train, param_name="C", param_range=np.logspace(-4, 1, 6), scoring='accuracy'
)
plt.title('5-fold validation for the hyperparameter C')
plt.savefig('LinealKernel_C_parameter_adult.pdf')
# %%
