import streamlit as st

import requests

import pandas as pd

import numpy as np

from datetime import datetime

from pathlib import Path

import os


# =========================================================

# PAGE CONFIGURATION

# =========================================================

st.set_page_config(

    page_title="AI Fraud Detection System",

    page_icon="🛡️",

    layout="wide",

    initial_sidebar_state="expanded"

)

# =========================================================

# CONFIGURATION

# =========================================================

API_BASE_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8001"
).rstrip("/")

API_URL = f"{API_BASE_URL}/predict"

BASE_DIR = Path(__file__).resolve().parents[2]

HISTORY_FILE = BASE_DIR / "prediction_history.csv"

# =========================================================

# CUSTOM CSS

# =========================================================

st.markdown(

    """

    <style>

    .main-title {

        font-size: 38px;

        font-weight: 700;

        margin-bottom: 5px;

    }

    .subtitle {

        font-size: 17px;

        color: #888888;

        margin-bottom: 25px;

    }

    .risk-low {

        padding: 15px;

        border-radius: 10px;

        background-color: #d4edda;

        color: #155724;

        font-weight: bold;

        text-align: center;

    }

    .risk-medium {

        padding: 15px;

        border-radius: 10px;

        background-color: #fff3cd;

        color: #856404;

        font-weight: bold;

        text-align: center;

    }

    .risk-high {

        padding: 15px;

        border-radius: 10px;

        background-color: #f8d7da;

        color: #721c24;

        font-weight: bold;

        text-align: center;

    }

    </style>

    """,

    unsafe_allow_html=True

)

# =========================================================

# SESSION STATE

# =========================================================

if "last_result" not in st.session_state:

    st.session_state.last_result = None

if "last_transaction" not in st.session_state:

    st.session_state.last_transaction = None

# =========================================================

# HISTORY FILE

# =========================================================

def load_history():

    if HISTORY_FILE.exists():

        try:

            df = pd.read_csv(HISTORY_FILE)

            return df

        except Exception:

            return pd.DataFrame()

    return pd.DataFrame()

def save_history(result, transaction):

    record = {

        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "transaction_id": result.get(

            "transaction_id",

            ""

        ),

        "prediction_id": result.get(

            "prediction_id",

            ""

        ),

        "amount": transaction["amount"],

        "fraud_probability": result.get(

            "fraud_probability",

            0

        ),

        "risk_score": result.get(

            "risk_score",

            0

        ),

        "risk_level": result.get(

            "risk_level",

            ""

        ),

        "action": result.get(

            "action",

            ""

        ),

        "model": result.get(

            "model",

            "XGBoost"

        ),

        "database_saved": result.get(

            "database_saved",

            False

        )

    }

    history = load_history()

    new_row = pd.DataFrame([record])

    history = pd.concat(

        [history, new_row],

        ignore_index=True

    )

    history.to_csv(

        HISTORY_FILE,

        index=False

    )

# =========================================================

# API CONNECTION TEST

# =========================================================

def check_api():
    try:
        response = requests.get(
            f"{API_BASE_URL}/",
            timeout=3
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# =========================================================

# HEADER

# =========================================================

st.markdown(

    '<div class="main-title">🛡️ AI Fraud Detection System</div>',

    unsafe_allow_html=True

)

st.markdown(

    '<div class="subtitle">XGBoost Based Fraud Detection & Risk Scoring Platform</div>',

    unsafe_allow_html=True

)

# =========================================================

# SIDEBAR

# =========================================================

st.sidebar.title("⚙️ System")

api_status = check_api()

if api_status:

    st.sidebar.success(

        "🟢 FastAPI Connected"

    )

else:

    st.sidebar.error(

        "🔴 FastAPI Offline"

    )

st.sidebar.markdown("---")

st.sidebar.info(

    """

    **Architecture**

    Transaction

    ↓

    Feature Engineering

    ↓

    XGBoost

    ↓

    Fraud Probability

    ↓

    Risk Score

    ↓

    PostgreSQL

    """

)

# =========================================================

# TABS

# =========================================================

tab1, tab2, tab3 = st.tabs(

    [

        "🔍 Fraud Detection",

        "📊 Analytics",

        "📋 History"

    ]

)

# =========================================================

# TAB 1 — FRAUD DETECTION

# =========================================================

with tab1:

    st.subheader(

        "💳 Transaction Analysis"

    )

    st.write(

        "Enter transaction information and analyze fraud risk."

    )

    # -----------------------------------------------------

    # BASIC TRANSACTION DETAILS

    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        time_value = st.number_input(

            "Transaction Time",

            min_value=0.0,

            value=0.0,

            step=1.0

        )

    with col2:

        amount = st.number_input(

            "Transaction Amount",

            min_value=0.0,

            value=100.0,

            step=10.0

        )

    with col3:

        transaction_hour = st.number_input(

            "Transaction Hour",

            min_value=0,

            max_value=23,

            value=12,

            step=1

        )

    is_night = st.checkbox(

        "🌙 Is Night Transaction?",

        value=False

    )

    st.markdown("---")

    # -----------------------------------------------------

    # V1 - V28

    # -----------------------------------------------------

    st.subheader(

        "🧮 Transaction Features"

    )

    st.caption(

        "V1–V28 are PCA-transformed transaction features used by the trained XGBoost model."

    )

    v_values = {}

    cols = st.columns(4)

    for i in range(1, 29):

        col_index = (i - 1) % 4

        with cols[col_index]:

            v_values[f"v{i}"] = st.number_input(

                f"V{i}",

                value=0.0,

                step=0.01,

                format="%.4f",

                key=f"v{i}"

            )

    st.markdown("---")

    # -----------------------------------------------------

    # CREATE TRANSACTION DATA

    # -----------------------------------------------------

    transaction_data = {

        "time": time_value,

        "amount": amount,

        "transaction_hour": int(

            transaction_hour

        ),

        "is_night": bool(

            is_night

        )

    }

    for i in range(1, 29):

        transaction_data[f"v{i}"] = v_values[

            f"v{i}"

        ]

    # -----------------------------------------------------

    # DETECT BUTTON

    # -----------------------------------------------------

    detect_button = st.button(

        "🚨 Detect Fraud",

        type="primary",

        use_container_width=True

    )

    if detect_button:

        if not api_status:

            st.error(

                "FastAPI is not running. Start FastAPI first on port 8001."

            )

        else:

            try:

                with st.spinner(

                    "Analyzing transaction..."

                ):

                    response = requests.post(
                    
                    API_URL,
                    
                    json=transaction_data,
                    
                    headers={
                    
                    "X-API-Key": os.getenv("FRAUD_API_KEY", "FraudDevKey_ChangeThis_2026")
                    
                    },
                    
                    timeout=30
                    
                    )

                # -------------------------------------------------

                # SUCCESS

                # -------------------------------------------------

                if response.status_code == 200:

                    result = response.json()

                    st.session_state.last_result = result

                    st.session_state.last_transaction = transaction_data

                    # Save locally for dashboard history

                    try:

                        save_history(

                            result,

                            transaction_data

                        )

                    except Exception as history_error:

                        st.warning(

                            f"Prediction received, but local history could not be saved: {history_error}"

                        )

                    st.success(

                        "✅ Transaction analyzed successfully!"

                    )

                    # -------------------------------------------------

                    # RESULT CARDS

                    # -------------------------------------------------

                    st.subheader(

                        "🎯 Fraud Detection Result"

                    )

                    c1, c2, c3, c4 = st.columns(4)

                    fraud_probability = float(

                        result.get(

                            "fraud_probability",

                            0

                        )

                    )

                    risk_score = float(

                        result.get(

                            "risk_score",

                            0

                        )

                    )

                    risk_level = result.get(

                        "risk_level",

                        "UNKNOWN"

                    )

                    action = result.get(

                        "action",

                        "UNKNOWN"

                    )

                    with c1:

                        st.metric(

                            "Fraud Probability",

                            f"{fraud_probability * 100:.2f}%"

                        )

                    with c2:

                        st.metric(

                            "Risk Score",

                            f"{risk_score:.2f}/100"

                        )

                    with c3:

                        st.metric(

                            "Risk Level",

                            risk_level

                        )

                    with c4:

                        st.metric(

                            "Action",

                            action

                        )

                    # -------------------------------------------------

                    # RISK MESSAGE

                    # -------------------------------------------------

                    if risk_level == "HIGH":

                        st.markdown(

                            '<div class="risk-high">🚨 HIGH RISK — TRANSACTION SHOULD BE BLOCKED</div>',

                            unsafe_allow_html=True

                        )

                    elif risk_level == "MEDIUM":

                        st.markdown(

                            '<div class="risk-medium">⚠️ MEDIUM RISK — TRANSACTION REQUIRES REVIEW</div>',

                            unsafe_allow_html=True

                        )

                    else:

                        st.markdown(

                            '<div class="risk-low">✅ LOW RISK — TRANSACTION CAN BE ALLOWED</div>',

                            unsafe_allow_html=True

                        )

                    st.markdown("---")

                    # -------------------------------------------------

                    # DATABASE INFORMATION

                    # -------------------------------------------------

                    st.subheader(

                        "🗄️ Database Information"

                    )

                    d1, d2, d3 = st.columns(3)

                    with d1:

                        st.metric(

                            "Transaction ID",

                            result.get(

                                "transaction_id",

                                "N/A"

                            )

                        )

                    with d2:

                        st.metric(

                            "Prediction ID",

                            result.get(

                                "prediction_id",

                                "N/A"

                            )

                        )

                    with d3:

                        database_saved = result.get(

                            "database_saved",

                            False

                        )

                        if database_saved:

                            st.success(

                                "PostgreSQL Saved"

                            )

                        else:

                            st.error(

                                "Database Not Saved"

                            )

                    # -------------------------------------------------

                    # MODEL INFORMATION

                    # -------------------------------------------------

                    st.markdown("---")

                    st.subheader(

                        "🤖 Model Information"

                    )

                    st.write(

                        f"**Model:** {result.get('model', 'XGBoost')}"

                    )

                    st.write(

                        f"**Features Used:** {result.get('features_used', 32)}"

                    )

                # -------------------------------------------------

                # ERROR

                # -------------------------------------------------

                else:

                    st.error(

                        f"FastAPI Error: {response.status_code}"

                    )

                    try:

                        error_data = response.json()

                        st.code(

                            str(error_data)

                        )

                    except Exception:

                        st.code(

                            response.text

                        )

            except requests.exceptions.ConnectionError:

                st.error(

                    "❌ Cannot connect to FastAPI. Make sure port 8001 server is running."

                )

            except requests.exceptions.Timeout:

                st.error(

                    "❌ FastAPI request timed out."

                )

            except Exception as e:

                st.error(

                    f"❌ Unexpected error: {e}"

                )

    # =====================================================

    # LAST RESULT

    # =====================================================

    if (

        st.session_state.last_result is not None

        and not detect_button

    ):

        result = st.session_state.last_result

        st.markdown("---")

        st.subheader(

            "📌 Last Prediction"

        )

        c1, c2, c3, c4 = st.columns(4)

        probability = float(

            result.get(

                "fraud_probability",

                0

            )

        )

        with c1:

            st.metric(

                "Fraud Probability",

                f"{probability * 100:.2f}%"

            )

        with c2:

            st.metric(

                "Risk Score",

                f"{result.get('risk_score', 0):.2f}"

            )

        with c3:

            st.metric(

                "Risk Level",

                result.get(

                    "risk_level",

                    "UNKNOWN"

                )

            )

        with c4:

            st.metric(

                "Action",

                result.get(

                    "action",

                    "UNKNOWN"

                )

            )

# =========================================================

# TAB 2 — ANALYTICS

# =========================================================

with tab2:

    st.subheader(

        "📊 Fraud Detection Analytics"

    )

    history = load_history()

    if history.empty:

        st.info(

            "No prediction history available yet. Perform some transactions first."

        )

    else:

        # -------------------------------------------------

        # DATA CLEANING

        # -------------------------------------------------

        if "fraud_probability" in history.columns:

            history["fraud_probability"] = pd.to_numeric(

                history["fraud_probability"],

                errors="coerce"

            )

        if "risk_score" in history.columns:

            history["risk_score"] = pd.to_numeric(

                history["risk_score"],

                errors="coerce"

            )

        if "amount" in history.columns:

            history["amount"] = pd.to_numeric(

                history["amount"],

                errors="coerce"

            )

        # -------------------------------------------------

        # METRICS

        # -------------------------------------------------

        total_transactions = len(

            history

        )

        potential_fraud = len(

            history[

                history["risk_level"].isin(

                    ["HIGH", "MEDIUM"]

                )

            ]

        )

        high_risk = len(

            history[

                history["risk_level"] == "HIGH"

            ]

        )

        fraud_rate = (

            potential_fraud / total_transactions * 100

            if total_transactions > 0

            else 0

        )

        avg_risk = (

            history["risk_score"].mean()

            if "risk_score" in history.columns

            else 0

        )

        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:

            st.metric(

                "Total Transactions",

                total_transactions

            )

        with m2:

            st.metric(

                "Potential Fraud",

                potential_fraud

            )

        with m3:

            st.metric(

                "High Risk",

                high_risk

            )

        with m4:

            st.metric(

                "Fraud Rate",

                f"{fraud_rate:.2f}%"

            )

        with m5:

            st.metric(

                "Average Risk",

                f"{avg_risk:.2f}"

            )

        st.markdown("---")

        # -------------------------------------------------

        # RISK DISTRIBUTION

        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(

                "Risk Distribution"

            )

            risk_distribution = (

                history["risk_level"]

                .value_counts()

            )

            st.bar_chart(

                risk_distribution

            )

        # -------------------------------------------------

        # FRAUD PROBABILITY TREND

        # -------------------------------------------------

        with col2:

            st.subheader(

                "Fraud Probability Trend"

            )

            trend_df = history[

                [

                    "timestamp",

                    "fraud_probability"

                ]

            ].copy()

            trend_df["timestamp"] = pd.to_datetime(

                trend_df["timestamp"],

                errors="coerce"

            )

            trend_df = trend_df.dropna()

            if not trend_df.empty:

                trend_df = trend_df.set_index(

                    "timestamp"

                )

                trend_df[

                    "fraud_probability"

                ] = (

                    trend_df[

                        "fraud_probability"

                    ] * 100

                )

                st.line_chart(

                    trend_df[

                        "fraud_probability"

                    ]

                )

        st.markdown("---")

        # -------------------------------------------------

        # TRANSACTION AMOUNT TREND

        # -------------------------------------------------

        st.subheader(

            "💰 Transaction Amount Trend"

        )

        amount_df = history[

            [

                "timestamp",

                "amount"

            ]

        ].copy()

        amount_df["timestamp"] = pd.to_datetime(

            amount_df["timestamp"],

            errors="coerce"

        )

        amount_df = amount_df.dropna()

        if not amount_df.empty:

            amount_df = amount_df.set_index(

                "timestamp"

            )

            st.line_chart(

                amount_df["amount"]

            )

        st.markdown("---")

        # -------------------------------------------------

        # HIGH RISK TRANSACTIONS

        # -------------------------------------------------

        st.subheader(

            "🚨 High Risk Transactions"

        )

        high_risk_df = history[

            history["risk_level"] == "HIGH"

        ]

        if high_risk_df.empty:

            st.success(

                "No HIGH risk transactions detected."

            )

        else:

            st.dataframe(

                high_risk_df,

                use_container_width=True

            )

# =========================================================

# TAB 3 — HISTORY

# =========================================================

with tab3:

    st.subheader(

        "📋 Prediction History"

    )

    history = load_history()

    if history.empty:

        st.info(

            "No prediction history available."

        )

    else:

        st.dataframe(

            history,

            use_container_width=True,

            height=500

        )

        st.markdown("---")

        # -------------------------------------------------

        # DOWNLOAD CSV

        # -------------------------------------------------

        csv_data = history.to_csv(

            index=False

        ).encode(

            "utf-8"

        )

        st.download_button(

            label="⬇️ Download Prediction History",

            data=csv_data,

            file_name="fraud_prediction_history.csv",

            mime="text/csv",

            use_container_width=True

        )

        # -------------------------------------------------

        # CLEAR HISTORY

        # -------------------------------------------------

        st.markdown("---")

        if st.button(

            "🗑️ Clear Local Prediction History"

        ):

            try:

                HISTORY_FILE.unlink(

                    missing_ok=True

                )

                st.success(

                    "Local prediction history cleared."

                )

                st.rerun()

            except Exception as e:

                st.error(

                    f"Could not clear history: {e}"

                )

# =========================================================

# FOOTER

# =========================================================

st.markdown("---")

st.caption(

    "AI-Powered Fraud Detection & Risk Scoring System | XGBoost + FastAPI + PostgreSQL + Streamlit"

)
