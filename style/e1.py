import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load and preprocess the dataset
# Replace with your data (X: features, y: labels)
X = np.random.rand(100, 5)  # Example feature set
y = np.random.randint(0, 3, 100)  # Example labels (3 classes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Train the LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Step 3: Evaluate the model
y_pred = lda.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 4: Hyperparameter tuning using GridSearchCV
param_grid = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]  # Shrinkage options for 'lsqr' and 'eigen'
}

lda_tuned = GridSearchCV(
    LinearDiscriminantAnalysis(),
    param_grid,
    scoring='accuracy',
    cv=5
)
lda_tuned.fit(X_train, y_train)

# Best parameters and accuracy
print("\nBest Parameters:", lda_tuned.best_params_)
print("Best Cross-Validation Accuracy:", lda_tuned.best_score_)

# Step 5: Evaluate the tuned model
y_pred_tuned = lda_tuned.predict(X_test)
print("\nTuned Model Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("Confusion Matrix for Tuned Model:\n", confusion_matrix(y_test, y_pred_tuned))
print("\nClassification Report for Tuned Model:\n", classification_report(y_test, y_pred_tuned))
