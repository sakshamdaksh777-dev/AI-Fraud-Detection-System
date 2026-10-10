import mlflow
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = list(
    BASE_DIR.rglob("xgboost_fraud_model.pkl")
)

DATA_FILES = list(
    BASE_DIR.rglob("creditcard.csv")
)


if not MODEL_FILES:
    raise FileNotFoundError(
        "xgboost_fraud_model.pkl not found"
    )

if not DATA_FILES:
    raise FileNotFoundError(
        "creditcard.csv not found"
    )


MODEL_FILE = MODEL_FILES[0]
DATA_FILE = DATA_FILES[0]


print("Model:", MODEL_FILE)
print("Dataset:", DATA_FILE)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading model...")

model = joblib.load(MODEL_FILE)

print("Model loaded:", type(model).__name__)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

df["Transaction_Hour"] = (
    (df["Time"] / 3600) % 24
).astype(int)


df["Is_Night"] = (
    (df["Transaction_Hour"] < 6) |
    (df["Transaction_Hour"] >= 22)
).astype(int)


# =========================================================
# FEATURE LIST
# =========================================================

feature_names = [
    "Time",

    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",

    "Amount",
    "Transaction_Hour",
    "Is_Night"
]


# =========================================================
# CHECK FEATURES
# =========================================================

missing_features = [
    feature
    for feature in feature_names
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        f"Missing features: {missing_features}"
    )


X = df[feature_names]

y = df["Class"]


print("\nFeatures:", X.shape)
print("Target:", y.shape)

print("\nFraud distribution:")
print(y.value_counts())


# =========================================================
# MODEL PREDICTION
# =========================================================

print("\nRunning predictions...")

y_probability = model.predict_proba(X)[:, 1]

y_prediction = (
    y_probability >= 0.50
).astype(int)


# =========================================================
# METRICS
# =========================================================

precision = precision_score(
    y,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y,
    y_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y,
    y_probability
)


print("\n====================================")
print("MODEL EVALUATION")
print("====================================")

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print("====================================")


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y,
    y_prediction
)

print("\nConfusion Matrix:")
print(cm)


# =========================================================
# SAVE CONFUSION MATRIX IMAGE
# =========================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(cm)

plt.title(
    "XGBoost Fraud Detection - Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

plt.yticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


CONFUSION_FILE = (
    BASE_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    CONFUSION_FILE,
    dpi=150
)

plt.close()


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    y,
    y_prediction,
    zero_division=0
)

print("\nClassification Report:")
print(report)


REPORT_FILE = (
    BASE_DIR /
    "classification_report.txt"
)

with open(
    REPORT_FILE,
    "w"
) as f:

    f.write(report)


# =========================================================
# MLFLOW
# =========================================================

print("\nConnecting to MLflow...")

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "AI-Fraud-Detection"
)


# =========================================================
# START RUN
# =========================================================

with mlflow.start_run(
    run_name="XGBoost-Evaluation"
):

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )


    # -----------------------------------------------------
    # PARAMETERS
    # -----------------------------------------------------

    mlflow.log_param(
        "model",
        "XGBoost"
    )

    mlflow.log_param(
        "features",
        32
    )

    mlflow.log_param(
        "classification_threshold",
        0.50
    )

    mlflow.log_param(
        "dataset",
        "creditcard.csv"
    )


    # -----------------------------------------------------
    # TAGS
    # -----------------------------------------------------

    mlflow.set_tag(
        "project",
        "AI Fraud Detection & Risk Scoring System"
    )

    mlflow.set_tag(
        "evaluation",
        "Fraud Detection"
    )

    mlflow.set_tag(
        "model_type",
        "XGBoost"
    )


    # -----------------------------------------------------
    # ARTIFACTS
    # -----------------------------------------------------

    mlflow.log_artifact(
        str(CONFUSION_FILE)
    )

    mlflow.log_artifact(
        str(REPORT_FILE)
    )


    # -----------------------------------------------------
    # RUN ID
    # -----------------------------------------------------

    run_id = (
        mlflow.active_run()
        .info
        .run_id
    )


    print("\n====================================")
    print("MLFLOW METRICS LOGGED SUCCESSFULLY")
    print("====================================")

    print(
        "Run ID:",
        run_id
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall:",
        round(recall, 4)
    )

    print(
        "F1 Score:",
        round(f1, 4)
    )

    print(
        "ROC-AUC:",
        round(roc_auc, 4)
    )

    print(
        "Confusion Matrix:",
        CONFUSION_FILE
    )

    print("====================================")