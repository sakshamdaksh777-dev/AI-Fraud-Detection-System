import mlflow
import mlflow.xgboost
import joblib
from pathlib import Path


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# FIND XGBOOST MODEL AUTOMATICALLY
# =========================================================

print("\nSearching for XGBoost model...")

MODEL_FILES = list(
    BASE_DIR.rglob("xgboost_fraud_model.pkl")
)


if not MODEL_FILES:

    print("\n❌ XGBoost model not found!")
    print("Search location:", BASE_DIR)

    raise FileNotFoundError(
        "xgboost_fraud_model.pkl not found anywhere in project."
    )


MODEL_FILE = MODEL_FILES[0]

print("Model found at:")
print(MODEL_FILE)


# =========================================================
# MLFLOW CONFIGURATION
# =========================================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "AI-Fraud-Detection"
)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading XGBoost model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")
print("Model type:", type(model).__name__)


# =========================================================
# START MLFLOW RUN
# =========================================================

with mlflow.start_run(
    run_name="XGBoost-Fraud-Detection"
):

    print("\nStarting MLflow run...")


    # =====================================================
    # PARAMETERS
    # =====================================================

    mlflow.log_param(
        "model",
        "XGBoost"
    )

    mlflow.log_param(
        "features",
        32
    )

    mlflow.log_param(
        "algorithm",
        "XGBoost Classifier"
    )

    mlflow.log_param(
        "risk_threshold_high",
        0.80
    )

    mlflow.log_param(
        "risk_threshold_medium",
        0.50
    )

    mlflow.log_param(
        "database",
        "PostgreSQL"
    )

    mlflow.log_param(
        "api",
        "FastAPI"
    )

    mlflow.log_param(
        "dashboard",
        "Streamlit"
    )


    # =====================================================
    # XGBOOST PARAMETERS
    # =====================================================

    try:

        if hasattr(model, "get_params"):

            params = model.get_params()

            parameter_names = [
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree"
            ]

            for key in parameter_names:

                if key in params:

                    value = params[key]

                    if value is not None:

                        mlflow.log_param(
                            key,
                            value
                        )

    except Exception as e:

        print(
            "Could not log XGBoost parameters:",
            e
        )


    # =====================================================
    # LOG XGBOOST MODEL
    # =====================================================

    print("\nLogging XGBoost model to MLflow...")

    mlflow.xgboost.log_model(
        model,
        artifact_path="xgboost-fraud-model"
    )


    # =====================================================
    # TAGS
    # =====================================================

    mlflow.set_tag(
        "project",
        "AI Fraud Detection & Risk Scoring System"
    )

    mlflow.set_tag(
        "model_type",
        "XGBoost"
    )

    mlflow.set_tag(
        "deployment",
        "FastAPI + Streamlit"
    )

    mlflow.set_tag(
        "database",
        "PostgreSQL"
    )

    mlflow.set_tag(
        "status",
        "Production Ready"
    )


    # =====================================================
    # FINISHED
    # =====================================================

    run_id = mlflow.active_run().info.run_id


    print("\n")
    print("==========================================")
    print("   MLFLOW RUN COMPLETED SUCCESSFULLY")
    print("==========================================")

    print("Run ID:", run_id)

    print("Experiment: AI-Fraud-Detection")

    print("Model: XGBoost")

    print("Features: 32")

    print("Model artifact: xgboost-fraud-model")

    print("==========================================")