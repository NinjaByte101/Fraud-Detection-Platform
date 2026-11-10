import os
import sqlite3
import time
from contextlib import closing

import pandas as pd
import streamlit as st

PRED_DB = "data/predictions.db"
CASE_DB = "data/cases.db"
STATUSES = ["new", "investigating", "closed"]
LABELS = ["unreviewed", "confirmed fraud", "false alarm"]

st.set_page_config(page_title="Fraud analyst dashboard", layout="wide")
st.title("Fraud analyst dashboard")


def load_predictions():
    if not os.path.exists(PRED_DB):
        return None
    with closing(sqlite3.connect(PRED_DB)) as db:
        df = pd.read_sql("SELECT * FROM predictions", db)
    # A transaction has no id in PaySim, so this combination identifies a case.
    df["case_id"] = (
        df["step"].astype(str) + "|" + df["nameOrig"] + "|" + df["nameDest"] + "|" + df["amount"].astype(str)
    )
    return df


def load_cases():
    with closing(sqlite3.connect(CASE_DB)) as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY, status TEXT, analyst_label TEXT,
                note TEXT, updated_at REAL)"""
        )
        db.commit()
        return pd.read_sql("SELECT * FROM cases", db)


def save_case(case_id, status, label, note):
    with closing(sqlite3.connect(CASE_DB)) as db:
        db.execute(
            "INSERT OR REPLACE INTO cases VALUES (?,?,?,?,?)",
            (case_id, status, label, note, time.time()),
        )
        db.commit()


predictions = load_predictions()
if predictions is None or predictions.empty:
    st.error("No predictions found. Run the Kafka replay first (data/predictions.db).")
    st.stop()

cases = load_cases()
alerts = predictions[predictions["predicted_fraud"] == 1].merge(cases, on="case_id", how="left")
alerts["status"] = alerts["status"].fillna("new")
alerts["analyst_label"] = alerts["analyst_label"].fillna("unreviewed")
alerts["note"] = alerts["note"].fillna("")
alerts = alerts.sort_values("fraud_probability", ascending=False)

# ---- Summary ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions scored", f"{len(predictions):,}")
c2.metric("Alerts", f"{len(alerts):,}", f"{len(alerts) / len(predictions):.2%} of scored")
c3.metric("Scoring latency p95", f"{predictions['service_latency_ms'].quantile(0.95):.1f} ms")
c4.metric("Open cases", int((alerts["status"] != "closed").sum()))

# ---- Feedback-based precision ----
reviewed = alerts[alerts["analyst_label"] != "unreviewed"]
confirmed = int((reviewed["analyst_label"] == "confirmed fraud").sum())
st.subheader("Analyst feedback")
f1, f2, f3 = st.columns(3)
f1.metric("Reviewed alerts", len(reviewed))
f2.metric("Confirmed fraud", confirmed)
f3.metric(
    "Precision from feedback",
    f"{confirmed / len(reviewed):.1%}" if len(reviewed) else "no labels yet",
)

# ---- Alert queue ----
st.subheader("Alert queue")
left, right = st.columns(2)
status_filter = left.multiselect("Status", STATUSES, default=["new", "investigating"])
min_prob = right.slider("Minimum fraud probability", 0.0, 1.0, 0.0, 0.01)
show_truth = st.checkbox("Show ground truth (demo only, an analyst would not have this)")

queue = alerts[alerts["status"].isin(status_filter) & (alerts["fraud_probability"] >= min_prob)]
columns = ["step", "type", "amount", "nameOrig", "nameDest", "fraud_probability", "status", "analyst_label"]
if show_truth:
    columns.append("true_label")
st.dataframe(queue[columns].head(500), use_container_width=True, hide_index=True)

if show_truth:
    precision = alerts["true_label"].mean()
    recall = alerts["true_label"].sum() / max(predictions["true_label"].sum(), 1)
    st.caption(f"Against ground truth: precision {precision:.1%}, recall {recall:.1%} on this replay.")

# ---- Review a case ----
st.subheader("Review a case")
if queue.empty:
    st.info("No alerts match the filters.")
else:
    options = queue["case_id"].head(200).tolist()
    labels = {
        row.case_id: f"{row.fraud_probability:.3f} | step {row.step} | {row.type} | {row.amount:,.0f} | {row.status}"
        for row in queue.head(200).itertuples()
    }
    chosen = st.selectbox("Case", options, format_func=lambda cid: labels[cid])
    current = queue[queue["case_id"] == chosen].iloc[0]
    st.write(
        f"**{current['nameOrig']}** to **{current['nameDest']}**, amount {current['amount']:,.2f}, "
        f"sender balance before {current['oldbalanceOrg']:,.2f}, receiver balance before {current['oldbalanceDest']:,.2f}"
    )
    with st.form("review"):
        status = st.selectbox("Status", STATUSES, index=STATUSES.index(current["status"]))
        label = st.selectbox("Analyst label", LABELS, index=LABELS.index(current["analyst_label"]))
        note = st.text_input("Note", value=current["note"])
        if st.form_submit_button("Save"):
            save_case(chosen, status, label, note)
            st.success("Saved")
            st.rerun()