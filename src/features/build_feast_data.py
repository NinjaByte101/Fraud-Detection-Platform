import pandas as pd

from src.features.build_features import RISKY_TYPES

DATA_PATH = "data/paysim.csv"
OUT_PATH = "data/account_stats.parquet"
BASE_TIME = pd.Timestamp("2026-01-01")  # PaySim has no dates, so step 1 = 1 hour after this
DEMO_START_STEP = 400


def build_account_stats(full: pd.DataFrame) -> pd.DataFrame:
    """Cumulative send/receive history per account, stamped one hour after each step."""
    late = full[full["type"].isin(RISKY_TYPES) & (full["step"] >= DEMO_START_STEP)]
    accounts = pd.Index(pd.concat([late["nameOrig"], late["nameDest"]]).unique())

    sent = (
        full[full["nameOrig"].isin(accounts)]
        .groupby(["nameOrig", "step"])
        .size()
        .rename("sent_n")
        .reset_index()
        .rename(columns={"nameOrig": "account_id"})
    )
    received = (
        full[full["nameDest"].isin(accounts)]
        .groupby(["nameDest", "step"])
        .agg(received_n=("amount", "size"), received_amt=("amount", "sum"))
        .reset_index()
        .rename(columns={"nameDest": "account_id"})
    )

    stats = (
        sent.merge(received, on=["account_id", "step"], how="outer")
        .fillna(0)
        .sort_values(["account_id", "step"])
    )
    for col in ["sent_n", "received_n", "received_amt"]:
        stats[col] = stats.groupby("account_id")[col].cumsum()

    stats = stats.rename(columns={
        "sent_n": "sent_count",
        "received_n": "received_count",
        "received_amt": "received_amount",
    })
    stats[["sent_count", "received_count"]] = stats[["sent_count", "received_count"]].astype("int64")
    stats["event_timestamp"] = BASE_TIME + pd.to_timedelta(stats["step"] + 1, unit="h")
    return stats[["account_id", "event_timestamp", "sent_count", "received_count", "received_amount"]]


if __name__ == "__main__":
    full = pd.read_csv(DATA_PATH)
    stats = build_account_stats(full)
    stats.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(stats):,} rows for {stats['account_id'].nunique():,} accounts")