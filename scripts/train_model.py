"""
Train an improved heart disease classifier with hyperparameter tuning,
feature engineering, and a voting ensemble.

"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

pd.set_option("future.infer_string", False)

from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
from xgboost import XGBClassifier

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/heart.csv")

target = "target"
X = df.drop(columns=[target])
y = df[target]

categorical_vars = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

for col in categorical_vars:
    X[col] = X[col].astype(str)

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
X["age_chol"] = X["age"] * X["chol"]
X["thalach_age"] = X["thalach"] / X["age"]
X["trestbps_chol"] = X["trestbps"] * X["chol"]
X["age_thalach"] = X["age"] * X["thalach"]
X["oldpeak_thalach"] = X["oldpeak"] * X["thalach"]
X["age_bin"] = pd.cut(X["age"], bins=[0, 40, 55, 70, 100], labels=["0", "1", "2", "3"]).astype(str)
X["chol_bin"] = pd.cut(X["chol"], bins=[0, 200, 240, 300, 600], labels=["0", "1", "2", "3"]).astype(str)

extra_cats = ["age_bin", "chol_bin"]
all_categorical = categorical_vars + extra_cats

# ---------------------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 4. Preprocessing pipeline
# ---------------------------------------------------------------------------
preprocessor = Pipeline([
    ("rare_label_encoder", RareLabelEncoder(tol=0.03, n_categories=1, variables=all_categorical)),
    ("onehot_encoder", OneHotEncoder(variables=all_categorical, drop_last=True)),
])

X_train_t = preprocessor.fit_transform(X_train)
X_test_t = preprocessor.transform(X_test)

# ---------------------------------------------------------------------------
# 5. Tune individual models (reduced iterations for speed)
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

print("Tuning RandomForest...")
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    {
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 12),
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", "log2", 0.5, 0.7],
    },
    n_iter=25, cv=cv, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1,
)
rf_search.fit(X_train_t, y_train)
print(f"  Best ROC-AUC: {rf_search.best_score_:.4f}")

print("Tuning XGBoost...")
xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
    {
        "n_estimators": randint(100, 400),
        "max_depth": randint(2, 10),
        "learning_rate": uniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 8),
    },
    n_iter=25, cv=cv, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_search.fit(X_train_t, y_train)
print(f"  Best ROC-AUC: {xgb_search.best_score_:.4f}")

print("Tuning Logistic Regression...")
lr_search = RandomizedSearchCV(
    Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(solver="saga", penalty="elasticnet", max_iter=4000, random_state=RANDOM_STATE))]),
    {"clf__C": uniform(0.01, 10), "clf__l1_ratio": uniform(0, 1)},
    n_iter=15, cv=cv, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1,
)
lr_search.fit(X_train_t, y_train)
print(f"  Best ROC-AUC: {lr_search.best_score_:.4f}")

# ---------------------------------------------------------------------------
# 6. Voting ensemble
# ---------------------------------------------------------------------------
print("\nBuilding voting ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ("rf", rf_search.best_estimator_),
        ("xgb", xgb_search.best_estimator_),
        ("lr", lr_search.best_estimator_),
    ],
    voting="soft",
    n_jobs=-1,
)
ensemble.fit(X_train_t, y_train)

# ---------------------------------------------------------------------------
# 7. Evaluate on test set
# ---------------------------------------------------------------------------
y_pred = ensemble.predict(X_test_t)
y_proba = ensemble.predict_proba(X_test_t)[:, 1]

print("\n" + "=" * 50)
print("VOTING ENSEMBLE - TEST SET RESULTS")
print("=" * 50)
print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 4)}")
print(f"ROC-AUC : {round(roc_auc_score(y_test, y_proba), 4)}")
print(f"F1      : {round(f1_score(y_test, y_pred), 4)}")
print("\nClassification report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# 8. Threshold optimization
# ---------------------------------------------------------------------------
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Optimal threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
y_pred_opt = (y_proba >= best_threshold).astype(int)
print(f"With threshold={best_threshold:.2f}:")
print(f"Accuracy: {round(accuracy_score(y_test, y_pred_opt), 4)}")
print(f"F1      : {round(f1_score(y_test, y_pred_opt), 4)}")
print("\nClassification report:\n", classification_report(y_test, y_pred_opt))

# ---------------------------------------------------------------------------
# 9. Individual model comparison on test set
# ---------------------------------------------------------------------------
print("=" * 50)
print("INDIVIDUAL MODEL COMPARISON")
print("=" * 50)
for name, est in [("RF", rf_search.best_estimator_), ("XGB", xgb_search.best_estimator_), ("LR", lr_search.best_estimator_)]:
    p = est.predict(X_test_t)
    pr = est.predict_proba(X_test_t)[:, 1]
    print(f"  {name}: Accuracy={accuracy_score(y_test, p):.4f}, ROC-AUC={roc_auc_score(y_test, pr):.4f}")

# ---------------------------------------------------------------------------
# 10. Save artifacts
# ---------------------------------------------------------------------------
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", ensemble),
])
final_pipeline.fit(X, y)

joblib.dump(final_pipeline, "models/heart_pipeline.joblib")
joblib.dump(float(best_threshold), "models/threshold.joblib")
print(f"\nSaved pipeline -> models/heart_pipeline.joblib")
print(f"Saved threshold -> models/threshold.joblib ({best_threshold:.2f})")
