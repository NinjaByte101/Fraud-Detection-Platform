import pandas as pd
from feast import FeatureStore

from src.features.build_feast_data import DEMO_START_STEP
from src.features.build_features import RISKY_TYPES, add_features, add_history_features
from src.features.verify_feast import feast_lookup

full = pd.read_csv("data/paysim.csv")
risky = add_features(full[full["type"].isin(RISKY_TYPES)].copy())
df = add_history_features(risky, full)
sample = df[df["step"] >= DEMO_START_STEP].sample(2000, random_state=0).reset_index(drop=True)

store = FeatureStore(repo_path="feature_repo")
dest = feast_lookup(store, sample["nameDest"], sample["step"])
sample["feast_received"] = dest["received_count"].values
bad = sample[sample["dest_prev_count"].fillna(0) != sample["feast_received"]].head(3)

stats = pd.read_parquet("data/account_stats.parquet")
for _, row in bad.iterrows():
    acct, step = row["nameDest"], row["step"]
    print("\n", acct, "| step", step, "| pandas:", row["dest_prev_count"], "| feast:", row["feast_received"])
    raw = full[(full["nameDest"] == acct) & (full["step"] < step)]
    print("raw transactions received before this step:", len(raw))
    print("steps:", sorted(raw["step"].unique())[:10])
    print(stats[stats["account_id"] == acct].sort_values("event_timestamp").head(8))