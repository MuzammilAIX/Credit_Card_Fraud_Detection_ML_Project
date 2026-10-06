
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.preprocessing import load_and_preprocess_data


# ============================================================
# 1. Project Paths
# ============================================================

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "credit_card.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgb_fraud_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)


# ============================================================
# 2. Create Models Folder
# ============================================================

os.makedirs(
    os.path.join(BASE_DIR, "models"),
    exist_ok=True
)


# ============================================================
# 3. Load and Preprocess Data
# ============================================================

print("\n===== Loading and Preprocessing Data =====")

print("Dataset path:", DATA_PATH)

X_train, X_test, y_train, y_test, scaler = (
    load_and_preprocess_data(DATA_PATH)
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("Training fraud distribution:")
print(y_train.value_counts())


# ============================================================
# 4. Logistic Regression
# ============================================================

model = LogisticRegression(
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)

y_pred_proba = model.predict_proba(
    X_test
)[:, 1]


print("\n===== Logistic Regression =====")

print(
    "Precision:",
    precision_score(y_test, y_pred)
)

print(
    "Recall:",
    recall_score(y_test, y_pred)
)

print(
    "F1 Score:",
    f1_score(y_test, y_pred)
)

print(
    "ROC AUC:",
    roc_auc_score(y_test, y_pred_proba)
)

print(
    "PR AUC:",
    average_precision_score(
        y_test,
        y_pred_proba
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 5. Logistic Regression + SMOTE
# ============================================================

print("\n===== Applying SMOTE =====")

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train,
        y_train
    )
)

print(
    "Original training shape:",
    X_train.shape
)

print(
    "SMOTE training shape:",
    X_train_smote.shape
)

print("\nSMOTE class distribution:")

print(
    y_train_smote.value_counts()
)


model_smote = LogisticRegression(
    random_state=42
)

model_smote.fit(
    X_train_smote,
    y_train_smote
)

y_pred_smote = model_smote.predict(
    X_test
)

y_proba_smote = model_smote.predict_proba(
    X_test
)[:, 1]


print("\n===== Logistic Regression + SMOTE =====")

print(
    "Precision:",
    precision_score(
        y_test,
        y_pred_smote
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        y_pred_smote
    )
)

print(
    "F1 Score:",
    f1_score(
        y_test,
        y_pred_smote
    )
)

print(
    "ROC AUC:",
    roc_auc_score(
        y_test,
        y_proba_smote
    )
)

print(
    "PR AUC:",
    average_precision_score(
        y_test,
        y_proba_smote
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_smote
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred_smote
    )
)


# ============================================================
# 6. Decision Tree
# ============================================================

dt_model = DecisionTreeClassifier(
    max_depth=15,
    class_weight="balanced",
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

y_pred_dt = dt_model.predict(
    X_test
)

y_proba_dt = dt_model.predict_proba(
    X_test
)[:, 1]


print("\n===== Decision Tree =====")

print(
    "Precision:",
    precision_score(
        y_test,
        y_pred_dt
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        y_pred_dt
    )
)

print(
    "F1 Score:",
    f1_score(
        y_test,
        y_pred_dt
    )
)

print(
    "ROC AUC:",
    roc_auc_score(
        y_test,
        y_proba_dt
    )
)

print(
    "PR AUC:",
    average_precision_score(
        y_test,
        y_proba_dt
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_dt
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred_dt
    )
)


# ============================================================
# 7. Random Forest
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

y_pred_rf = rf_model.predict(
    X_test
)

y_proba_rf = rf_model.predict_proba(
    X_test
)[:, 1]


print("\n===== Random Forest =====")

print(
    "Precision:",
    precision_score(
        y_test,
        y_pred_rf
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        y_pred_rf
    )
)

print(
    "F1 Score:",
    f1_score(
        y_test,
        y_pred_rf
    )
)

print(
    "ROC AUC:",
    roc_auc_score(
        y_test,
        y_proba_rf
    )
)

print(
    "PR AUC:",
    average_precision_score(
        y_test,
        y_proba_rf
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_rf
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred_rf
    )
)


# ============================================================
# 8. XGBoost
# ============================================================

scale_pos_weight = (
    (y_train == 0).sum()
    /
    (y_train == 1).sum()
)

print(
    "\nScale Pos Weight:",
    scale_pos_weight
)


xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train,
    y_train
)

y_pred_xgb = xgb_model.predict(
    X_test
)

y_proba_xgb = xgb_model.predict_proba(
    X_test
)[:, 1]


print("\n===== XGBoost =====")

print(
    "Precision:",
    precision_score(
        y_test,
        y_pred_xgb
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        y_pred_xgb
    )
)

print(
    "F1 Score:",
    f1_score(
        y_test,
        y_pred_xgb
    )
)

print(
    "ROC AUC:",
    roc_auc_score(
        y_test,
        y_proba_xgb
    )
)

print(
    "PR AUC:",
    average_precision_score(
        y_test,
        y_proba_xgb
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_xgb
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred_xgb
    )
)


# ============================================================
# 9. XGBoost Hyperparameter Tuning
# ============================================================

print("\n===== XGBoost Hyperparameter Tuning =====")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}


cv_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="average_precision",
    cv=5,
    n_jobs=-1,
    verbose=2
)


cv_xgb.fit(
    X_train,
    y_train
)


# ============================================================
# 10. Best XGBoost Model
# ============================================================

print("\n===== Best XGBoost Parameters =====")

print(
    cv_xgb.best_params_
)

print(
    "Best CV PR AUC:",
    cv_xgb.best_score_
)


best_xgb_model = (
    cv_xgb.best_estimator_
)


# ============================================================
# 11. Evaluate Tuned XGBoost
# ============================================================

y_pred_tuned_xgb = (
    best_xgb_model.predict(
        X_test
    )
)

y_proba_tuned_xgb = (
    best_xgb_model.predict_proba(
        X_test
    )[:, 1]
)


print("\n===== Tuned XGBoost =====")

print(
    "Precision:",
    precision_score(
        y_test,
        y_pred_tuned_xgb
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        y_pred_tuned_xgb
    )
)

print(
    "F1 Score:",
    f1_score(
        y_test,
        y_pred_tuned_xgb
    )
)

print(
    "ROC AUC:",
    roc_auc_score(
        y_test,
        y_proba_tuned_xgb
    )
)

print(
    "PR AUC:",
    average_precision_score(
        y_test,
        y_proba_tuned_xgb
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_tuned_xgb
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred_tuned_xgb
    )
)


# ============================================================
# 12. Feature Importance
# ============================================================

feature_names = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Hour", "Amount_log"]
)


feature_importance = (
    best_xgb_model.feature_importances_
)


print("\n===== Feature Importance =====")


for feature, importance in sorted(
    zip(
        feature_names,
        feature_importance
    ),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{feature}: {importance:.6f}"
    )


# ============================================================
# 13. Final Threshold
# ============================================================

y_proba_best_xgb = (
    best_xgb_model.predict_proba(
        X_test
    )[:, 1]
)


final_threshold = 0.5


y_pred_final = (
    y_proba_best_xgb >= final_threshold
).astype(int)


# ============================================================
# 14. Final Evaluation
# ============================================================

print("\n===== Final XGBoost Evaluation =====")

print(
    "Final Threshold:",
    final_threshold
)

print(
    "Precision:",
    precision_score(
        y_test,
        y_pred_final
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        y_pred_final
    )
)

print(
    "F1 Score:",
    f1_score(
        y_test,
        y_pred_final
    )
)

print(
    "ROC AUC:",
    roc_auc_score(
        y_test,
        y_proba_best_xgb
    )
)

print(
    "PR AUC:",
    average_precision_score(
        y_test,
        y_proba_best_xgb
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_final
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred_final
    )
)


# ============================================================
# 15. Save Final Model
# ============================================================

joblib.dump(
    best_xgb_model,
    MODEL_PATH
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)


# ============================================================
# 16. Save Scaler
# ============================================================

joblib.dump(
    scaler,
    SCALER_PATH
)

print(
    f"Scaler saved to: {SCALER_PATH}"
)


print("\n===== Training Completed Successfully =====")
