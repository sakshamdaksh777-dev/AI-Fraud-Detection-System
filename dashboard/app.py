import streamlit as st
import requests


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🛡️ AI Fraud Detection System")
st.write("AI-powered transaction fraud detection and risk scoring")


# ============================================================
# API CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8001/predict"


# ============================================================
# TRANSACTION INPUT
# ============================================================

st.subheader("Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:

    time = st.number_input(
        "Time",
        value=0.0
    )

    amount = st.number_input(
        "Amount",
        value=100.0,
        min_value=0.0
    )

    transaction_hour = st.number_input(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=12
    )

with col2:

    is_night = st.selectbox(
        "Is Night?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col3:

    st.info(
        "Enter the transaction's V1–V28 "
        "feature values below."
    )


# ============================================================
# V1 - V28
# ============================================================

st.subheader("Transaction Features")

values = {}

columns = st.columns(4)

for i in range(1, 29):

    with columns[(i - 1) % 4]:

        values[f"v{i}"] = st.number_input(
            f"V{i}",
            value=0.0,
            format="%.6f",
            key=f"v{i}"
        )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Detect Fraud",
    type="primary",
    use_container_width=True
)


# ============================================================
# API REQUEST
# ============================================================

if predict_button:

    payload = {

        "time": time,
        "amount": amount,

        "v1": values["v1"],
        "v2": values["v2"],
        "v3": values["v3"],
        "v4": values["v4"],
        "v5": values["v5"],
        "v6": values["v6"],
        "v7": values["v7"],
        "v8": values["v8"],
        "v9": values["v9"],
        "v10": values["v10"],
        "v11": values["v11"],
        "v12": values["v12"],
        "v13": values["v13"],
        "v14": values["v14"],
        "v15": values["v15"],
        "v16": values["v16"],
        "v17": values["v17"],
        "v18": values["v18"],
        "v19": values["v19"],
        "v20": values["v20"],
        "v21": values["v21"],
        "v22": values["v22"],
        "v23": values["v23"],
        "v24": values["v24"],
        "v25": values["v25"],
        "v26": values["v26"],
        "v27": values["v27"],
        "v28": values["v28"],

        "transaction_hour": transaction_hour,
        "is_night": is_night
    }


    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )


        # ====================================================
        # SUCCESS
        # ====================================================

        if response.status_code == 200:

            result = response.json()

            st.success("Transaction analyzed successfully!")


            # ------------------------------------------------
            # RESULT CARDS
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Fraud Probability",
                    f"{result['fraud_probability'] * 100:.2f}%"
                )


            with col2:

                st.metric(
                    "Risk Score",
                    f"{result['risk_score']:.2f}"
                )


            with col3:

                st.metric(
                    "Risk Level",
                    result["risk_level"]
                )


            with col4:

                st.metric(
                    "Action",
                    result["action"]
                )


            # ------------------------------------------------
            # RISK MESSAGE
            # ------------------------------------------------

            if result["risk_level"] == "HIGH":

                st.error(
                    "🚨 HIGH RISK TRANSACTION — BLOCK"
                )

            elif result["risk_level"] == "MEDIUM":

                st.warning(
                    "⚠️ MEDIUM RISK TRANSACTION — REVIEW"
                )

            else:

                st.success(
                    "✅ LOW RISK TRANSACTION — ALLOW"
                )


            # ------------------------------------------------
            # RAW RESPONSE
            # ------------------------------------------------

            with st.expander("View API Response"):

                st.json(result)


        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.code(response.text)


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to Fraud Detection API."
        )

        st.info(
            "Make sure FastAPI is running on port 8001."
        )


    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )