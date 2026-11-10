import argparse
import json
import sqlite3
import time
import uuid

import numpy as np
import requests
from confluent_kafka import Consumer

TOPIC = "transactions"
SCORE_URL = "http://localhost:8000/score"
DB_PATH = "data/predictions.db"
SERVICE_FIELDS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest"]


def open_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            step INTEGER, type TEXT, amount REAL,
            nameOrig TEXT, oldbalanceOrg REAL,
            nameDest TEXT, oldbalanceDest REAL,
            fraud_probability REAL, predicted_fraud INTEGER, true_label INTEGER,
            service_latency_ms REAL, end_to_end_ms REAL, scored_at REAL
        )
    """)
    return db


def main(max_messages: int):
    # A fresh group id means each run starts from "now", not from old messages.
    consumer = Consumer({
        "bootstrap.servers": "localhost:9092",
        "group.id": f"scorer-{uuid.uuid4().hex[:8]}",
        "auto.offset.reset": "latest",
    })
    consumer.subscribe([TOPIC], on_assign=lambda c, parts: print("Ready - start the producer now"))

    db = open_db()
    session = requests.Session()
    seen, scored, caught, service_ms, e2e_ms = 0, 0, 0, [], []
    last_message = None  # the 20 s silence timer only starts after the first message

    while seen < max_messages and (last_message is None or time.time() - last_message < 20):
        msg = consumer.poll(1.0)
        if msg is None or msg.error():
            continue
        last_message = time.time()
        seen += 1

        tx = json.loads(msg.value())
        reply = session.post(SCORE_URL, json={k: tx[k] for k in SERVICE_FIELDS}).json()
        if not reply["scored"]:
            continue

        now = time.time()
        end_to_end = (now - tx["sent_at"]) * 1000
        scored += 1
        caught += int(reply["is_fraud"] and tx["isFraud"] == 1)
        service_ms.append(reply["latency_ms"])
        e2e_ms.append(end_to_end)

        db.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (tx["step"], tx["type"], tx["amount"], tx["nameOrig"], tx["oldbalanceOrg"],
             tx["nameDest"], tx["oldbalanceDest"], reply["fraud_probability"],
             int(reply["is_fraud"]), tx["isFraud"], reply["latency_ms"], end_to_end, now),
        )
        if scored % 200 == 0:
            db.commit()

    db.commit()
    consumer.close()

    print(f"\nRead {seen:,} messages, scored {scored:,}")
    for name, values in [("scoring service", service_ms), ("end to end (Kafka + API)", e2e_ms)]:
        if values:
            p50, p95, p99 = np.percentile(values, [50, 95, 99])
            print(f"{name}: p50 {p50:.1f} ms | p95 {p95:.1f} ms | p99 {p99:.1f} ms")
    print(f"Fraud caught among scored: {caught}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=5000)
    args = parser.parse_args()
    main(args.max)