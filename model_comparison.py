import time
from pathlib import Path

import mlflow
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

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

DATA_FILES = list(
    BASE_DIR.rglob("creditcard.csv")
)

if not DATA_FILES:
    raise FileNotFoundError(
        "creditcard.csv not found"
    )

DATA_FILE = DATA_FILES[0]

print("Dataset:", DATA_FILE)


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
# TRAIN / TEST SPLIT
# =========================================================

print("\nCreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# =========================================================
# MLFLOW
# =========================================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "AI-Fraud-Detection"
)


# =========================================================
# MODELS
# =========================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# =========================================================
# RESULTS
# =========================================================

results = []


# =========================================================
# TRAIN MODELS
# =========================================================

for model_name, model in models.items():

    print("\n")
    print("========================================")
    print("Training:", model_name)
    print("========================================")

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start_time
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    y_probability = (
        model.predict_proba(X_test)[:, 1]
    )

    y_prediction = (
        y_probability >= 0.50
    ).astype(int)


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    precision = precision_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    # -----------------------------------------------------
    # PRINT
    # -----------------------------------------------------

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

    print(
        f"Training Time: {training_time:.2f} sec"
    )


    # -----------------------------------------------------
    # SAVE RESULT
    # -----------------------------------------------------

    results.append({

        "Model": model_name,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc,

        "Training Time": training_time

    })


    # =====================================================
    # MLFLOW RUN
    # =====================================================

    with mlflow.start_run(
        run_name=f"Comparison-{model_name}"
    ):

        # Parameters

        mlflow.log_param(
            "model",
            model_name
        )

        mlflow.log_param(
            "features",
            32
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "classification_threshold",
            0.50
        )


        # Metrics

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

        mlflow.log_metric(
            "training_time_seconds",
            training_time
        )


        # Tags

        mlflow.set_tag(
            "project",
            "AI Fraud Detection"
        )

        mlflow.set_tag(
            "comparison",
            "Logistic Regression vs Random Forest vs XGBoost"
        )


# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)


print("\n\n")
print("========================================")
print("MODEL COMPARISON RESULTS")
print("========================================")

print(
    results_df.to_string(
        index=False
    )
)


# =========================================================
# SAVE RESULTS
# =========================================================

RESULT_FILE = (
    BASE_DIR /
    "model_comparison_results.csv"
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# =========================================================
# BEST MODEL
# =========================================================

best_model = results_df.iloc[0]

print("\n")
print("========================================")
print("BEST MODEL")
print("========================================")

print(
    "Model:",
    best_model["Model"]
)

print(
    "F1 Score:",
    round(
        best_model["F1 Score"],
        4
    )
)

print(
    "ROC-AUC:",
    round(
        best_model["ROC-AUC"],
        4
    )
)

print(
    "Precision:",
    round(
        best_model["Precision"],
        4
    )
)

print(
    "Recall:",
    round(
        best_model["Recall"],
        4
    )

)

print("========================================")

print(
    "\nResults saved to:",
    RESULT_FILE
)