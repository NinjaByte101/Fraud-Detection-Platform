import os

import pytest

needs_artifacts = pytest.mark.skipif(
    not (os.path.exists("models/xgb.json") and os.path.exists("feature_repo/data/online_store.db")),
    reason="needs the trained model and the Feast online store (see README, steps 3 and 4)",
)

FRAUD = {
    "step": 1, "type": "TRANSFER", "amount": 181.0,
    "nameOrig": "C1305486145", "oldbalanceOrg": 181.0,
    "nameDest": "C553264065", "oldbalanceDest": 0.0,
}
PAYMENT = {
    "step": 1, "type": "PAYMENT", "amount": 9839.64,
    "nameOrig": "C1231006815", "oldbalanceOrg": 170136.0,
    "nameDest": "M1979787155", "oldbalanceDest": 0.0,
}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.serving.app import app

    return TestClient(app)


@needs_artifacts
def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert 0 < body["threshold"] < 1


@needs_artifacts
def test_payment_is_not_scored(client):
    body = client.post("/score", json=PAYMENT).json()
    assert body["scored"] is False


@needs_artifacts
def test_known_fraud_transfer_is_flagged(client):
    body = client.post("/score", json=FRAUD).json()
    assert body["scored"] is True
    assert 0 <= body["fraud_probability"] <= 1
    assert body["is_fraud"] is True
    assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold"])
    assert body["latency_ms"] >= 0


@needs_artifacts
def test_incomplete_request_is_rejected(client):
    assert client.post("/score", json={"step": 1}).status_code == 422
