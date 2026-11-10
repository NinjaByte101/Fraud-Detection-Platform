import pandas as pd
from feast import FeatureStore

from src.features.build_feast_data import BASE_TIME, DEMO_START_STEP
from src.features.build_features import RISKY_TYPES, add_features, add_history_features

DATA_PATH = "data/paysim.csv"
FEATURES = [
    "account_stats:sent_count",
    "account_stats:received_count",
    "account_stats:received_amount",
]


def feast_lookup(store, accounts, steps):
    """Ask Feast what it knew about each account at the moment of each transaction."""
    keys = pd.DataFrame({
        "account_id": accounts.values,
        "event_timestamp": BASE_TIME + pd.to_timedelta(steps.values, unit="h"),
    })
    # Feast drops duplicate lookups, so ask once per unique (account, time) pair
    unique_keys = keys.drop_duplicates().reset_index(drop=True)
    found = store.get_historical_features(entity_df=unique_keys, features=FEATURES).to_df()
    found["event_timestamp"] = (
        pd.to_datetime(found["event_timestamp"], utc=True).dt.tz_localize(None)
    )
    # Join back by key, so every sample row gets its own answer in the original order
    merged = keys.merge(found, on=["account_id", "event_timestamp"], how="left")
    return merged.fillna(0)


def main():
    full = pd.read_csv(DATA_PATH)
    risky = add_features(full[full["type"].isin(RISKY_TYPES)].copy())
    with_history = add_history_features(risky, full)

    sample = (
        with_history[with_history["step"] >= DEMO_START_STEP]
        .sample(2000, random_state=0)
        .reset_index(drop=True)
    )

    store = FeatureStore(repo_path="feature_repo")
    orig = feast_lookup(store, sample["nameOrig"], sample["step"])
    dest = feast_lookup(store, sample["nameDest"], sample["step"])

    checks = {
        "orig_prev_count vs sent_count": (sample["orig_prev_count"], orig["sent_count"]),
        "dest_prev_count vs received_count": (sample["dest_prev_count"], dest["received_count"]),
        "dest_prev_amount vs received_amount": (sample["dest_prev_amount"], dest["received_amount"]),
    }
    for name, (pandas_values, feast_values) in checks.items():
        diff = (pandas_values.fillna(0) - feast_values).abs()
        print(f"{name}: {(diff > 1e-6).sum()} mismatches out of {len(diff)}, max diff {diff.max():.4f}")

    # Is the online store filled? (this is what the live API will read)
    account = sample.loc[0, "nameDest"]
    online = store.get_online_features(
        features=FEATURES, entity_rows=[{"account_id": account}]
    ).to_dict()
    print("online store lookup:", online)


if __name__ == "__main__":
    main()