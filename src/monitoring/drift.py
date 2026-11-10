import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

SAMPLE = 5000          # rows per window, keeps the report fast
SHARE_THRESHOLD = 0.3  # retrain if more than 30% of features drifted
FLAG_PATH = "data/retrain_flag.json"


def drift_check(reference: pd.DataFrame, current: pd.DataFrame, name: str) -> float:
    """Save an HTML report and return the share of drifted features."""
    definition = DataDefinition(numerical_columns=list(reference.columns))
    ref = Dataset.from_pandas(
        reference.sample(min(SAMPLE, len(reference)), random_state=0).reset_index(drop=True),
        data_definition=definition,
    )
    cur = Dataset.from_pandas(
        current.sample(min(SAMPLE, len(current)), random_state=0).reset_index(drop=True),
        data_definition=definition,
    )
    snapshot = Report([DataDriftPreset()]).run(cur, ref)

    os.makedirs("reports", exist_ok=True)
    snapshot.save_html(f"reports/drift_{name}.html")

    metrics = snapshot.dict()["metrics"]
    share = next(
        m["value"]["share"] for m in metrics if m["metric_name"].startswith("DriftedColumnsCount")
    )
    print(f"{name}: {share:.0%} of {len(reference.columns)} features drifted -> reports/drift_{name}.html")
    return float(share)


def maybe_retrain(share: float, name: str, train_cmd: list, run: bool):
    if share <= SHARE_THRESHOLD:
        print(f"  below threshold ({SHARE_THRESHOLD:.0%}), no retraining needed")
        return
    os.makedirs("data", exist_ok=True)
    with open(FLAG_PATH, "w") as f:
        json.dump({"dataset": name, "drift_share": share, "created": time.time()}, f)
    print(f"  ABOVE threshold, wrote {FLAG_PATH}")
    if run:
        print("  retraining:", " ".join(train_cmd))
        subprocess.run(train_cmd, check=True)


def paysim(run: bool):
    from src.features.build_features import add_features, load_risky_transactions

    cols = ["amount", "oldbalanceOrg", "oldbalanceDest", "amount_to_balance", "hour_of_day"]
    df = add_features(load_risky_transactions("data/paysim.csv"))
    share = drift_check(df[df["step"] < 350][cols], df[df["step"] >= 400][cols], "paysim_400plus")
    maybe_retrain(share, "paysim", [sys.executable, "-m", "src.models.train_xgb", "--features", "history"], run)


def elliptic(run: bool):
    from src.graph.load_elliptic import load_elliptic

    data = load_elliptic(local_only=True)
    x = pd.DataFrame(data["x"], columns=[f"f{i}" for i in range(data["x"].shape[1])])
    steps = data["time_step"]
    reference = x[steps <= 29]
    worst = 0.0
    for label, mask in [("35_42", (steps >= 35) & (steps <= 42)), ("43_49", steps >= 43)]:
        worst = max(worst, drift_check(reference, x[mask], f"elliptic_{label}"))
    maybe_retrain(worst, "elliptic", [sys.executable, "-m", "src.graph.train_xgb_elliptic", "--local-only"], run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["paysim", "elliptic"])
    parser.add_argument("--run-retrain", action="store_true", help="actually run training when drift is high")
    args = parser.parse_args()
    {"paysim": paysim, "elliptic": elliptic}[args.dataset](args.run_retrain)