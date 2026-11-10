import pandas as pd

RISKY_TYPES = ["TRANSFER", "CASH_OUT"]

# Everything we can compute, including post-transaction balances (leaky).
FEATURE_COLS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorOrig",
    "errorDest",
    "emptiedOrig",
    "amount_to_balance",
    "hour_of_day",
    "is_transfer",
]

# Only what is known before the transaction completes (safe for real-time scoring).
SERVING_COLS = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_balance",
    "would_empty_orig",
    "dest_was_empty",
    "hour_of_day",
    "is_transfer",
]



FEATURE_SETS = {"all": FEATURE_COLS, "serving": SERVING_COLS}


HISTORY_COLS = [
    "orig_prev_count",
    "dest_prev_count",
    "dest_prev_amount",
]

FEATURE_SETS["history"] = SERVING_COLS + HISTORY_COLS


def add_history_features(risky: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Count what each account did in earlier steps (never the current step)."""
    out = risky.copy()

    for role, name_col in [("orig", "nameOrig"), ("dest", "nameDest")]:
        per_step = (
            full.groupby([name_col, "step"])
            .agg(n=("amount", "size"), amt=("amount", "sum"))
            .reset_index()
            .sort_values([name_col, "step"])
        )
        # cumulative total minus this step's own part = strictly earlier steps
        per_step[f"{role}_prev_count"] = (
            per_step.groupby(name_col)["n"].cumsum() - per_step["n"]
        )
        per_step[f"{role}_prev_amount"] = (
            per_step.groupby(name_col)["amt"].cumsum() - per_step["amt"]
        )
        keep = [name_col, "step", f"{role}_prev_count", f"{role}_prev_amount"]
        out = out.merge(per_step[keep], on=[name_col, "step"], how="left")

    return out
def load_risky_transactions(path: str) -> pd.DataFrame:
    """Load PaySim and keep only the types that can contain fraud."""
    df = pd.read_csv(path)
    return df[df["type"].isin(RISKY_TYPES)].copy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model features. Uses only fields known at transaction time."""
    df = df.copy()

    # Does the sender's balance change match the amount sent?
    df["errorOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Was the sender's account emptied by this transaction?
    df["emptiedOrig"] = (
        (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
    ).astype(int)

    # How big is the amount compared to what the sender had?
    df["amount_to_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    df["hour_of_day"] = df["step"] % 24
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)

        # Would this transaction drain the sender's account? (known before it happens)
    df["would_empty_orig"] = (
        (df["amount"] >= df["oldbalanceOrg"]) & (df["oldbalanceOrg"] > 0)
    ).astype(int)

    # Is the receiver an empty account? (new or throwaway accounts are suspicious)
    df["dest_was_empty"] = (df["oldbalanceDest"] == 0).astype(int)
    return df


def time_split(df: pd.DataFrame, valid_start: int = 350, test_start: int = 400):
    """Train on early steps, validate on the next block, test on the latest."""
    train = df[df["step"] < valid_start]
    valid = df[(df["step"] >= valid_start) & (df["step"] < test_start)]
    test = df[df["step"] >= test_start]
    return train, valid, test