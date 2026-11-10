import sqlite3

import numpy as np
import pandas as pd

db = sqlite3.connect("data/predictions.db")
df = pd.read_sql("SELECT * FROM predictions", db)

tp = int(((df["predicted_fraud"] == 1) & (df["true_label"] == 1)).sum())
fp = int(((df["predicted_fraud"] == 1) & (df["true_label"] == 0)).sum())
fn = int(((df["predicted_fraud"] == 0) & (df["true_label"] == 1)).sum())

print(f"Scored: {len(df):,} | true fraud: {int(df['true_label'].sum())} | steps {df['step'].min()}-{df['step'].max()}")
print(f"TP {tp} | FP {fp} | FN {fn}")
print(f"Precision: {tp / max(tp + fp, 1):.3f} | Recall: {tp / max(tp + fn, 1):.3f}")

warm = df.iloc[200:]  # skip the startup backlog
p50, p95, p99 = np.percentile(warm["end_to_end_ms"], [50, 95, 99])
print(f"End to end without first 200 messages: p50 {p50:.1f} | p95 {p95:.1f} | p99 {p99:.1f} ms")