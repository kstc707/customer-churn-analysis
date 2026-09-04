"""
Customer Churn Prediction — Modeling Phase
Trains and evaluates Logistic Regression and Random Forest classifiers on the
IBM Telco Customer Churn dataset, building on the EDA in customer_churn_eda.ipynb.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score

# 1. Load and clean data (same steps as the EDA notebook)
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Logistic Regression (scaled features, class-weighted for imbalance)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=2000, class_weight="balanced")
log_model.fit(X_train_s, y_train)
log_pred = log_model.predict(X_test_s)
log_prob = log_model.predict_proba(X_test_s)[:, 1]

# 3. Random Forest (class-weighted)
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# 4. Evaluation
for name, y_pred, y_prob in [
    ("Logistic Regression", log_pred, log_prob),
    ("Random Forest", rf_pred, rf_prob),
]:
    print(f"=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))
    print("F1 (churn class):", round(f1_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred))

# 5. Feature importance (Random Forest)
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print("Top 10 predictive features:")
print(importances.sort_values(ascending=False).head(10))
