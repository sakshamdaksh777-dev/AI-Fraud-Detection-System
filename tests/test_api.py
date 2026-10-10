
from fastapi.testclient import TestClient
from dashboard.api.main import app

client = TestClient(app)


def test_api_docs_available():
    response = client.get("/docs")
    assert response.status_code == 200



def test_prediction_requires_api_key():
    response = client.post(
        "/predict",
        headers={"X-API-Key": "wrong-key"},
        json={
            "Time": 0,
            "Amount": 100,
            "Transaction_Hour": 12,
            "Is_Night": False,
            **{f"V{i}": 0 for i in range(1, 29)},
        },
    )
    assert response.status_code == 401


from dashboard.api.main import calculate_risk


def test_low_risk():
    score, level, action = calculate_risk(0.30)
    assert score == 30.0
    assert level == "LOW"
    assert action == "ALLOW"


def test_medium_risk():
    score, level, action = calculate_risk(0.70)
    assert score == 70.0
    assert level == "MEDIUM"
    assert action == "REVIEW"


def test_high_risk():
    score, level, action = calculate_risk(0.96)
    assert score == 96.0
    assert level == "HIGH"
    assert action == "BLOCK"
