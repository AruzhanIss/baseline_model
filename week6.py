import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

data_path = "/Users/arukaiss/Desktop/Новая папка/fraud_data.csv"
df = pd.read_csv(data_path)

df["amount_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_base = rf.predict(X_test)

print("\n Baseline random forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_base))
print("Precision:", precision_score(y_test, y_pred_base))
print("Recall:", recall_score(y_test, y_pred_base))
print("F1 score:", f1_score(y_test, y_pred_base))

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n Лучшие параметры:", grid_search.best_params_)

best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test)

print("\n Tuned random forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("Precision:", precision_score(y_test, y_pred_tuned))
print("Recall:", recall_score(y_test, y_pred_tuned))
print("F1 score:", f1_score(y_test, y_pred_tuned))

print("\nClassification report:\n", classification_report(y_test, y_pred_tuned))
