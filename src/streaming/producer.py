import argparse
import json
import time

import pandas as pd
from confluent_kafka import Producer

DATA_PATH = "data/paysim.csv"
TOPIC = "transactions"
REPLAY_START_STEP = 400  # the Feast online store holds history up to here


def main(limit: int, rate: float):
    df = pd.read_csv(DATA_PATH)
    df = df[df["step"] >= REPLAY_START_STEP].sort_values("step", kind="stable").head(limit)

    producer = Producer({"bootstrap.servers": "localhost:9092"})
    delay = 1.0 / rate if rate else 0

    for row in df.itertuples(index=False):
        message = {
            "step": int(row.step),
            "type": row.type,
            "amount": float(row.amount),
            "nameOrig": row.nameOrig,
            "oldbalanceOrg": float(row.oldbalanceOrg),
            "nameDest": row.nameDest,
            "oldbalanceDest": float(row.oldbalanceDest),
            "isFraud": int(row.isFraud),  # true label, kept for evaluation only
            "sent_at": time.time(),
        }
        producer.produce(TOPIC, key=row.nameOrig, value=json.dumps(message))
        producer.poll(0)
        if delay:
            time.sleep(delay)

    producer.flush()
    print(f"Sent {len(df):,} transactions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--rate", type=float, default=200, help="messages per second, 0 = as fast as possible")
    args = parser.parse_args()
    main(args.limit, args.rate)