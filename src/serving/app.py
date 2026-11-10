import json
import time
from collections import defaultdict

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from feast import FeatureStore
from pydantic import BaseModel

RISKY_TYPES = {"TRANSFER", "CASH_OUT"}
FEAST_FEATURES = [
    "account_stats:sent_count",
    "account_stats:received_count",
    "account_stats:received_amount",
]

app = FastAPI(title="Fraud scoring service")

model = xgb.XGBClassifier()
model.load_model("models/xgb.json")
with open("models/xgb_meta.json") as f:
    meta = json.load(f)
store = FeatureStore(repo_path="feature_repo")

# What this service has seen since it started (the online store only knows
# history up to step 400). Used to keep features point-in-time correct.
sent_log = defaultdict(list)       # account -> [step, ...]
received_log = defaultdict(list)   # account -> [(step, amount), ...]


class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    nameDest: str
    oldbalanceDest: float


def account_history(tx: Transaction) -> dict:
    """Past activity of sender and receiver, counting only earlier steps."""
    online = store.get_online_features(
        features=FEAST_FEATURES,
        entity_rows=[{"account_id": tx.nameOrig}, {"account_id": tx.nameDest}],
    ).to_dict()

    sent_before = sum(1 for s in sent_log[tx.nameOrig] if s < tx.step)
    received_before = [(s, a) for s, a in received_log[tx.nameDest] if s < tx.step]

    return {
        "orig_prev_count": (online["sent_count"][0] or 0) + sent_before,
        "dest_prev_count": (online["received_count"][1] or 0) + len(received_before),
        "dest_prev_amount": (online["received_amount"][1] or 0.0)
        + sum(a for _, a in received_before),
    }


def build_features(tx: Transaction) -> pd.DataFrame:
    row = {
        "amount": tx.amount,
        "oldbalanceOrg": tx.oldbalanceOrg,
        "oldbalanceDest": tx.oldbalanceDest,
        "amount_to_balance": tx.amount / (tx.oldbalanceOrg + 1),
        "would_empty_orig": int(tx.amount >= tx.oldbalanceOrg and tx.oldbalanceOrg > 0),
        "dest_was_empty": int(tx.oldbalanceDest == 0),
        "hour_of_day": tx.step % 24,
        "is_transfer": int(tx.type == "TRANSFER"),
        **account_history(tx),
    }
    return pd.DataFrame([row])[meta["features"]]


@app.get("/health")
def health():
    return {"status": "ok", "feature_set": meta["feature_set"], "threshold": meta["threshold"]}


@app.post("/score")
def score(tx: Transaction):
    started = time.perf_counter()
    result = {"scored": False}

    if tx.type in RISKY_TYPES:
        features = build_features(tx)
        probability = float(model.predict_proba(features)[0, 1])
        result = {
            "scored": True,
            "fraud_probability": round(probability, 6),
            "is_fraud": probability >= meta["threshold"],
            "threshold": meta["threshold"],
        }

    # Remember this transaction for later ones (after scoring it).
    sent_log[tx.nameOrig].append(tx.step)
    received_log[tx.nameDest].append((tx.step, tx.amount))

    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result