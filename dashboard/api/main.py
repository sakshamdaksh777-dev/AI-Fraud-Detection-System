# =========================================================
# AI FRAUD DETECTION SYSTEM - FASTAPI
# =========================================================
import os
import secrets

from fastapi import Header, Depends
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

from dashboard.api.database import save_transaction, save_prediction


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="AI Fraud Detection API",
    description="XGBoost Based Fraud Detection & Risk Scoring API",
    version="2.0.0"
)
# ==========================================
# API SECURITY CONFIGURATION
# ==========================================

API_KEY = os.getenv("FRAUD_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "FRAUD_API_KEY environment variable is not set."
    )


def verify_api_key(
    x_api_key: str = Header(...)
):
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    return True
# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "models"
    / "xgboost_fraud_model.pkl"
)

print("=" * 60)
print("LOADING FRAUD DETECTION MODEL")
print("=" * 60)

print("Model path:")
print(MODEL_FILE)

print("Model exists:", MODEL_FILE.exists())


try:

    model = joblib.load(MODEL_FILE)

    print("✅ XGBoost MODEL LOADED SUCCESSFULLY")

except Exception as e:

    print("❌ MODEL LOADING FAILED")
    print("ERROR:", e)

    model = None

print("=" * 60)

# =========================================================
# OPTIMIZED THRESHOLD
# =========================================================

# Obtained from threshold optimization
# Precision = 0.8989
# Recall    = 0.8163
# F1 Score  = 0.8556
# ROC-AUC   = 0.9813

FRAUD_THRESHOLD = 0.95


# =========================================================
# TRANSACTION INPUT SCHEMA
# =========================================================

class Transaction(BaseModel):

    # Original transaction time
    time: float

    # Transaction amount
    amount: float

    # PCA / transaction features
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float

    # Engineered features
    transaction_hour: int
    is_night: bool


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "AI Fraud Detection API is running",
        "model": "XGBoost",
        "optimized_threshold": FRAUD_THRESHOLD,
        "features": 32
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    if model is None:

        return {
            "status": "unhealthy",
            "model_loaded": False
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "XGBoost",
        "optimized_threshold": FRAUD_THRESHOLD
    }


# =========================================================
# RISK CALCULATION
# =========================================================

def calculate_risk(probability):

    risk_score = probability * 100

    # HIGH / BLOCK
    if probability >= FRAUD_THRESHOLD:

        risk_level = "HIGH"
        action = "BLOCK"

    # MEDIUM / REVIEW
    elif probability >= 0.50:

        risk_level = "MEDIUM"
        action = "REVIEW"

    # LOW / ALLOW
    else:

        risk_level = "LOW"
        action = "ALLOW"

    return risk_score, risk_level, action


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(transaction: Transaction):

    try:

        # -------------------------------------------------
        # CHECK MODEL
        # -------------------------------------------------

        if model is None:

            raise HTTPException(
                status_code=500,
                detail="Fraud detection model is not loaded."
            )


        # -------------------------------------------------
        # CREATE FEATURE DATA
        # -------------------------------------------------

        data = {

            "Time": transaction.time,

            "V1": transaction.v1,
            "V2": transaction.v2,
            "V3": transaction.v3,
            "V4": transaction.v4,
            "V5": transaction.v5,
            "V6": transaction.v6,
            "V7": transaction.v7,
            "V8": transaction.v8,
            "V9": transaction.v9,
            "V10": transaction.v10,
            "V11": transaction.v11,
            "V12": transaction.v12,
            "V13": transaction.v13,
            "V14": transaction.v14,
            "V15": transaction.v15,
            "V16": transaction.v16,
            "V17": transaction.v17,
            "V18": transaction.v18,
            "V19": transaction.v19,
            "V20": transaction.v20,
            "V21": transaction.v21,
            "V22": transaction.v22,
            "V23": transaction.v23,
            "V24": transaction.v24,
            "V25": transaction.v25,
            "V26": transaction.v26,
            "V27": transaction.v27,
            "V28": transaction.v28,

            "Amount": transaction.amount,

            "Transaction_Hour": transaction.transaction_hour,

            "Is_Night": int(transaction.is_night)
        }


        # -------------------------------------------------
        # EXACT MODEL FEATURE ORDER
        # -------------------------------------------------

        feature_order = [

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


        # -------------------------------------------------
        # CREATE DATAFRAME
        # -------------------------------------------------

        df = pd.DataFrame(
            [[data[column] for column in feature_order]],
            columns=feature_order
        )


        # -------------------------------------------------
        # VERIFY FEATURES
        # -------------------------------------------------

        if df.shape[1] != 32:

            raise ValueError(
                f"Feature mismatch. Expected 32, got {df.shape[1]}"
            )


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        # Ensure the input uses the exact feature names expected by XGBoost
        
        
        # MODEL PREDICTION
        # Feature order is already defined in feature_order
        df = df[feature_order].astype(float)

        # Avoid stale feature-name metadata from the saved XGBoost model
        model.get_booster().feature_names = None

        fraud_probability = float(
            model.predict_proba(df.to_numpy(dtype=float))[0][1]
        )





        # -------------------------------------------------
        # RISK CALCULATION
        # -------------------------------------------------

        risk_score, risk_level, action = calculate_risk(
            fraud_probability
        )


        # -------------------------------------------------
        # DATABASE TRANSACTION DATA
        # -------------------------------------------------

        transaction_data = {

            "time": transaction.time,

            "amount": transaction.amount,

            "v1": transaction.v1,
            "v2": transaction.v2,
            "v3": transaction.v3,
            "v4": transaction.v4,
            "v5": transaction.v5,
            "v6": transaction.v6,
            "v7": transaction.v7,
            "v8": transaction.v8,
            "v9": transaction.v9,
            "v10": transaction.v10,
            "v11": transaction.v11,
            "v12": transaction.v12,
            "v13": transaction.v13,
            "v14": transaction.v14,
            "v15": transaction.v15,
            "v16": transaction.v16,
            "v17": transaction.v17,
            "v18": transaction.v18,
            "v19": transaction.v19,
            "v20": transaction.v20,
            "v21": transaction.v21,
            "v22": transaction.v22,
            "v23": transaction.v23,
            "v24": transaction.v24,
            "v25": transaction.v25,
            "v26": transaction.v26,
            "v27": transaction.v27,
            "v28": transaction.v28,

            "transaction_hour": transaction.transaction_hour,

            "is_night": transaction.is_night
        }


        # -------------------------------------------------
        # SAVE TRANSACTION
        # -------------------------------------------------

        transaction_id = save_transaction(
            transaction_data
        )


        # -------------------------------------------------
        # SAVE PREDICTION
        # -------------------------------------------------

        prediction_id = save_prediction(

            transaction_id=transaction_id,

            fraud_probability=fraud_probability,

            risk_score=risk_score,

            risk_level=risk_level,

            action=action
        )


        # -------------------------------------------------
        # FINAL RESPONSE
        # -------------------------------------------------

        return {
    "fraud_probability": round(fraud_probability, 6),
    "risk_score": round(risk_score, 2),
    "risk_level": risk_level,
    "action": action,
    "model": "XGBoost",
    "features_used": 32,
    "optimized_threshold": FRAUD_THRESHOLD,
    "transaction_id": transaction_id,
    "prediction_id": prediction_id,
    "database_saved": True
}


    except HTTPException:

        raise


    except Exception as e:

        print("=" * 60)
        print("❌ PREDICTION ERROR")
        print(e)
        print("=" * 60)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )