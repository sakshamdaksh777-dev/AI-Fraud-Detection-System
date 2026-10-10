# AI Fraud Detection & Risk Scoring System

## Overview
An AI-powered fraud detection system that analyzes transaction features using an XGBoost classification model and assigns a fraud probability, risk score, risk level, and recommended action.

## Key Features
- XGBoost-based fraud prediction
- Fraud probability and risk score calculation
- LOW, MEDIUM, and HIGH risk classification
- ALLOW, REVIEW, and BLOCK actions
- FastAPI prediction API protected by API-key authentication
- Streamlit interactive dashboard
- PostgreSQL storage for transactions and predictions
- Automated tests using pytest

## Technology Stack
- Python
- XGBoost, Pandas, scikit-learn
- FastAPI
- Streamlit
- PostgreSQL
- SQLAlchemy
- pytest

## Project Workflow
1. User enters transaction details in the dashboard.
2. Streamlit sends the data to the FastAPI endpoint.
3. The API validates the input and API key.
4. The XGBoost model calculates the fraud probability.
5. The system calculates the risk score, risk level, and action.
6. Transaction details and prediction results are stored in PostgreSQL.
7. The dashboard displays the result.

## Run Locally

Activate the virtual environment in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Load environment variables from `.env` before starting the services.

Start the API in one terminal:

```powershell
python -m uvicorn dashboard.api.main:app --host 127.0.0.1 --port 8001
```

Start the dashboard in a second terminal:

```powershell
python -m streamlit run dashboard/dashboard/app.py --server.port 8503
```

Open the dashboard at http://localhost:8503.

API documentation is available at http://localhost:8001/docs.

## Run Tests

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); import pytest; raise SystemExit(pytest.main(['-v']))"
```

## Security Notes
- Keep `.env` out of version control.
- Use a strong, unique API key and database password.
- Do not commit credentials or real transaction data.

## Limitations
- Predictions depend on the quality and representativeness of the training dataset.
- A model prediction is a risk estimate, not proof of fraud.
- Risk thresholds should be validated against held-out data and business requirements.
- The current tests cover basic API behavior and risk classification; they do not establish model accuracy or production readiness.

