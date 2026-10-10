import numpy as np
import pandas as pd
import joblib
import mlflow

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================================================
# PROJECT PATH
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

print("\nLoading XGBoost model...")

model = joblib.load(MODEL_FILE)

print(
    "Model loaded:",
    type(model).__name__
)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(
    "Dataset shape:",
    df.shape
)


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
# FEATURES
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


X = df[feature_names]

y = df["Class"]


# =========================================================
# TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "\nTest samples:",
    len(X_test)
)


# =========================================================
# PREDICT FRAUD PROBABILITY
# =========================================================

print(
    "\nCalculating fraud probabilities..."
)

probabilities = model.predict_proba(
    X_test
)[:, 1]


roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print(
    "ROC-AUC:",
    round(roc_auc, 4)
)


# =========================================================
# THRESHOLD SEARCH
# =========================================================

thresholds = np.arange(
    0.05,
    0.96,
    0.01
)


results = []


print(
    "\nSearching for best threshold..."
)


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)


    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    results.append({

        "threshold": round(
            float(threshold),
            2
        ),

        "precision": precision,

        "recall": recall,

        "f1_score": f1

    })


results_df = pd.DataFrame(
    results
)


# =========================================================
# BEST F1 THRESHOLD
# =========================================================

best_row = results_df.loc[
    results_df["f1_score"].idxmax()
]


best_threshold = float(
    best_row["threshold"]
)

best_precision = float(
    best_row["precision"]
)

best_recall = float(
    best_row["recall"]
)

best_f1 = float(
    best_row["f1_score"]
)


# =========================================================
# DISPLAY RESULT
# =========================================================

print("\n")
print(
    "=============================================="
)

print(
    "       THRESHOLD OPTIMIZATION RESULT"
)

print(
    "=============================================="
)

print(
    "Best Threshold :",
    round(best_threshold, 2)
)

print(
    "Precision      :",
    round(best_precision, 4)
)

print(
    "Recall         :",
    round(best_recall, 4)
)

print(
    "F1 Score       :",
    round(best_f1, 4)
)

print(
    "ROC-AUC        :",
    round(roc_auc, 4)
)

print(
    "=============================================="
)


# =========================================================
# SAVE ALL THRESHOLD RESULTS
# =========================================================

RESULT_FILE = (
    BASE_DIR /
    "threshold_results.csv"
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)


print(
    "\nThreshold results saved:"
)

print(
    RESULT_FILE
)


# =========================================================
# RISK LEVEL LOGIC
# =========================================================

# Optimized threshold becomes the fraud decision
# threshold.
#
# We keep a higher threshold for HIGH risk.

HIGH_THRESHOLD = 0.80


def risk_level(probability):

    if probability >= HIGH_THRESHOLD:

        return "HIGH"

    elif probability >= best_threshold:

        return "MEDIUM"

    else:

        return "LOW"


def action_from_risk(level):

    if level == "HIGH":

        return "BLOCK"

    elif level == "MEDIUM":

        return "REVIEW"

    else:

        return "ALLOW"


# =========================================================
# SHOW SAMPLE RISK DECISIONS
# =========================================================

print("\n")
print(
    "=============================================="
)

print(
    "        RISK DECISION EXAMPLES"
)

print(
    "=============================================="
)


sample_probabilities = [
    0.02,
    0.20,
    best_threshold,
    0.60,
    0.85,
    0.95
]


for probability in sample_probabilities:

    level = risk_level(
        probability
    )

    action = action_from_risk(
        level
    )

    print(
        f"Probability: {probability:.2f} "
        f"-> {level} -> {action}"
    )


# =========================================================
# MLFLOW
# =========================================================

print(
    "\nLogging results to MLflow..."
)

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "AI-Fraud-Detection"
)


with mlflow.start_run(
    run_name="Threshold-Optimization"
):

    # -----------------------------------------------------
    # PARAMETERS
    # -----------------------------------------------------

    mlflow.log_param(
        "model",
        "XGBoost"
    )

    mlflow.log_param(
        "threshold_search_start",
        0.05
    )

    mlflow.log_param(
        "threshold_search_end",
        0.95
    )

    mlflow.log_param(
        "threshold_step",
        0.01
    )

    mlflow.log_param(
        "high_risk_threshold",
        HIGH_THRESHOLD
    )


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    mlflow.log_metric(
        "optimized_threshold",
        best_threshold
    )

    mlflow.log_metric(
        "optimized_precision",
        best_precision
    )

    mlflow.log_metric(
        "optimized_recall",
        best_recall
    )

    mlflow.log_metric(
        "optimized_f1",
        best_f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )


    # -----------------------------------------------------
    # ARTIFACT
    # -----------------------------------------------------

    mlflow.log_artifact(
        str(RESULT_FILE)
    )


    # -----------------------------------------------------
    # TAGS
    # -----------------------------------------------------

    mlflow.set_tag(
        "project",
        "AI Fraud Detection"
    )

    mlflow.set_tag(
        "optimization",
        "F1 Score Threshold Optimization"
    )

    mlflow.set_tag(
        "model",
        "XGBoost"
    )


    print(
        "\nMLflow logging successful!"
    )


# =========================================================
# FINAL
# =========================================================

print("\n")
print(
    "=============================================="
)

print(
    "       THRESHOLD OPTIMIZATION COMPLETE"
)

print(
    "=============================================="
)

print(
    "Best threshold:",
    round(best_threshold, 2)
)

print(
    "Saved file:",
    "threshold_results.csv"
)

print(
    "MLflow experiment:",
    "AI-Fraud-Detection"
)

print(
    "=============================================="
)