# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, ValidationCurveDisplay, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
import time

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

# Grid search for hyperparameter optimization
param_range = np.logspace(-3, 1, 5)
param_grid = {'C': param_range}
grid_search = GridSearchCV(SVC(kernel="linear"), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_C = grid_search.best_params_['C']
best_accuracy = grid_search.best_score_

print(f"Best parameter C: {best_C}")
print(f"Accuracy with best C: {best_accuracy:.4f}")

# Training the final model with best C
final_model = SVC(kernel='linear', C=best_C, probability=True)
final_model.fit(X_train, y_train)

# Evaluating the final model
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]  

print("\nTest Set Evaluation")
print(classification_report(y_test, y_pred, digits=3))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for SVM with Linear Kernel')
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
plt.show()
plt.savefig('LinearKernel_ROC_adult.pdf')

# %%
ValidationCurveDisplay.from_estimator(
   estimator=SVC(kernel='linear'), X=X_train, y=y_train, param_name="C", param_range=np.logspace(-6, 3, 5), scoring='accuracy'
)

best_svm = SVC(kernel='linear', C=best_C)

# Define learning curve parameters
train_sizes = np.linspace(0.1, 1.0, 5)

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

plt.title(f"Learning curve for SVM with Linear kernel")
plt.savefig('LinearKernel_learning_adult.pdf')
# %%
