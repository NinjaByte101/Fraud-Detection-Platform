import pandas as pd

RISKY_TYPES = ["TRANSFER", "CASH_OUT"]

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
    return df


def time_split(df: pd.DataFrame, split_step: int = 600):
    """Train on early steps, test on later steps."""
    train = df[df["step"] <= split_step]
    test = df[df["step"] > split_step]
    return train, test