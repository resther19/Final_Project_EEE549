from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
import torch
def cancer():

    #%% load Wisconsin Breast Cancer dataset using ucimlrepo
    # fetch dataset 
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 

    # data (as pandas dataframes) 
    X= breast_cancer_wisconsin_diagnostic.data.features 
    y = breast_cancer_wisconsin_diagnostic.data.targets 
    y['Diagnosis'] = y['Diagnosis'].map({'M': 1, 'B': 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.LongTensor(y_train['Diagnosis'].values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.LongTensor(y_test['Diagnosis'].values)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
