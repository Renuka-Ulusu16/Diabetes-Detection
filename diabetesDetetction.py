import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

# Load dataset
data = pd.read_csv("diabetes.csv")

# Replace 0s in specific columns with NaN
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)

# Impute missing values
imputer = SimpleImputer(strategy="median")
data[cols_with_zeros] = imputer.fit_transform(data[cols_with_zeros])

# Features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(probability=True)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", round(auc, 3))
    results[name] = {"model": model, "accuracy": acc, "roc_auc": auc}

# Plot ROC curves
plt.figure(figsize=(10, 7))
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result["model"].predict_proba(X_test_scaled)[:, 1])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Feature importance (Random Forest)
rf_model = results["Random Forest"]["model"]
feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importance.sort_values().plot(kind="barh", title="Feature Importance (Random Forest)", figsize=(8, 5))
plt.tight_layout()
plt.show()

# Save the best model
best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])[1]['model']
joblib.dump(best_model, "diabetes_best_model.pkl")
joblib.dump(scaler, "diabetes_scaler.pkl")

print("\n🔒 Model and scaler saved successfully!")

