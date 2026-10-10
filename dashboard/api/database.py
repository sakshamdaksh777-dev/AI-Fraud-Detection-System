from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# =========================================================# =========================================================
# AI FRAUD DETECTION SYSTEM - DATABASE CONFIGURATION
# PostgreSQL + SQLAlchemy
# =========================================================

import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# =========================================================
# DATABASE CONFIGURATION
# ================================================
DB_HOST = os.getenv("DB_HOST", "db")
DB_NAME = os.getenv("DB_NAME", "fraud_detection")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = int(os.getenv("DB_PORT", "5432"))


if not DB_PASSWORD:
    raise RuntimeError(
        "DB_PASSWORD environment variable is not set."
    )


# =========================================================
# DATABASE URL
# =========================================================

DATABASE_URL = URL.create(
    drivername="postgresql+pg8000",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)


# =========================================================
# CREATE DATABASE ENGINE
# =========================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# =========================================================
# TEST DATABASE CONNECTION
# =========================================================

def test_connection():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        return result.scalar()


# =========================================================
# SAVE TRANSACTION
# =========================================================

def save_transaction(data):

    query = text("""
        INSERT INTO transactions (
            transaction_time,
            amount,
            v1, v2, v3, v4, v5, v6, v7, v8,
            v9, v10, v11, v12, v13, v14, v15, v16,
            v17, v18, v19, v20, v21, v22, v23, v24,
            v25, v26, v27, v28,
            transaction_hour,
            is_night
        )
        VALUES (
            :time,
            :amount,
            :v1, :v2, :v3, :v4, :v5, :v6, :v7, :v8,
            :v9, :v10, :v11, :v12, :v13, :v14, :v15, :v16,
            :v17, :v18, :v19, :v20, :v21, :v22, :v23, :v24,
            :v25, :v26, :v27, :v28,
            :transaction_hour,
            :is_night
        )
        RETURNING id
    """)

    with engine.begin() as connection:
        result = connection.execute(query, data)
        return result.scalar()


# =========================================================
# SAVE FRAUD PREDICTION
# =========================================================

def save_prediction(
    transaction_id,
    fraud_probability,
    risk_score,
    risk_level,
    action,
):

    query = text("""
        INSERT INTO predictions (
            transaction_id,
            fraud_probability,
            risk_score,
            risk_level,
            action,
            model_name
        )
        VALUES (
            :transaction_id,
            :fraud_probability,
            :risk_score,
            :risk_level,
            :action,
            :model_name
        )
        RETURNING id
    """)

    values = {
        "transaction_id": transaction_id,
        "fraud_probability": fraud_probability,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "action": action,
        "model_name": "XGBoost",
    }

    with engine.begin() as connection:
        result = connection.execute(query, values)
        return result.scalar()


# =========================================================
# RUN DATABASE CONNECTION TEST
# =========================================================

if __name__ == "__main__":

    print("Testing PostgreSQL connection...")

    try:
        result = test_connection()
        print("Database connection successful!")
        print("Result:", result)

    except Exception as error:
        print("Database connection failed!")
        print(error)
# DATABASE CONFIG
# =========================================================

DB_HOST = "localhost"
DB_NAME = "fraud_detection"
DB_USER = "postgres"

# YAHAN APNA ACTUAL POSTGRESQL PASSWORD DALO
DB_PASSWORD = "skm#daksh"

DB_PORT = 5432


# =========================================================
# CREATE DATABASE URL
# =========================================================

DATABASE_URL = URL.create(
    drivername="postgresql+pg8000",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME
)


# =========================================================
# CREATE ENGINE
# =========================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# =========================================================
# TEST CONNECTION
# =========================================================

def test_connection():

    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT 1")
        )

        return result.scalar()


# =========================================================
# SAVE TRANSACTION
# =========================================================

def save_transaction(data):

    query = text("""
        INSERT INTO transactions (
            transaction_time,
            amount,
            v1, v2, v3, v4, v5, v6, v7, v8,
            v9, v10, v11, v12, v13, v14, v15, v16,
            v17, v18, v19, v20, v21, v22, v23, v24,
            v25, v26, v27, v28,
            transaction_hour,
            is_night
        )

        VALUES (
            :time,
            :amount,
            :v1, :v2, :v3, :v4, :v5, :v6, :v7, :v8,
            :v9, :v10, :v11, :v12, :v13, :v14, :v15, :v16,
            :v17, :v18, :v19, :v20, :v21, :v22, :v23, :v24,
            :v25, :v26, :v27, :v28,
            :transaction_hour,
            :is_night
        )

        RETURNING id
    """)

    with engine.begin() as connection:

        result = connection.execute(
            query,
            data
        )

        return result.scalar()


# =========================================================
# SAVE PREDICTION
# =========================================================

def save_prediction(
    transaction_id,
    fraud_probability,
    risk_score,
    risk_level,
    action
):

    query = text("""
        INSERT INTO predictions (
            transaction_id,
            fraud_probability,
            risk_score,
            risk_level,
            action,
            model_name
        )

        VALUES (
            :transaction_id,
            :fraud_probability,
            :risk_score,
            :risk_level,
            :action,
            :model_name
        )

        RETURNING id
    """)

    values = {
        "transaction_id": transaction_id,
        "fraud_probability": fraud_probability,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "action": action,
        "model_name": "XGBoost"
    }

    with engine.begin() as connection:

        result = connection.execute(
            query,
            values
        )

        return result.scalar()


# =========================================================
# RUN DATABASE TEST
# =========================================================

if __name__ == "__main__":

    print("Testing PostgreSQL connection...")

    try:

        result = test_connection()

        print("Database connection successful!")
        print("Result:", result)

    except Exception as e:

        print("Database connection failed!")
        print(e)