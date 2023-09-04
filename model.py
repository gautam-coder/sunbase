import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load the provided dataset and perform initial data exploration
data = pd.read_excel('customer_churn_large_dataset.xlsx')

# Drop non-informative columns if needed
data = data.drop(['CustomerID', 'Name'], axis=1)

# Handle missing data
data = data.dropna()

# Encode categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['Gender', 'Location'])

# Relevant Features
data['Monthly_Cost'] = data['Monthly_Bill'] * data['Total_Usage_GB']
data['Age_Monthly_Bill'] = data['Age'] * data['Monthly_Bill']

# Split the data into features (X) and target (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Address Class Imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

# Model Evaluation
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

import joblib
joblib.dump(best_model, 'best_model.pkl')

joblib.dump(scaler, 'scaler.pkl')
