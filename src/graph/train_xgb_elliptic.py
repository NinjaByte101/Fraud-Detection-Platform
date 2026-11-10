import argparse

import mlflow
import numpy as np
import xgboost as xgb

from src.graph.load_elliptic import load_elliptic
from src.graph.metrics import best_f1_threshold, evaluate, print_results


def main(local_only: bool = False):
    data = load_elliptic(local_only=local_only)
    x, y, steps = data["x"], data["y"], data["time_step"]
    train, valid, test = data["train_mask"], data["valid_mask"], data["test_mask"]

    pos_weight = (y[train] == 0).sum() / (y[train] == 1).sum()
    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(pos_weight),
        "eval_metric": "aucpr",
        "early_stopping_rounds": 30,
        "tree_method": "hist",
        "random_state": 42,
    }

    run_name = "xgb-local-only" if local_only else "xgb-node-features"
    mlflow.set_experiment("fraud-elliptic")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("local_only", local_only)

        model = xgb.XGBClassifier(**params)
        model.fit(x[train], y[train], eval_set=[(x[valid], y[valid])], verbose=50)

        scores = model.predict_proba(x)[:, 1]
        threshold = best_f1_threshold(y[valid], scores[valid])
        results = evaluate(y, scores, steps, test, threshold)

        mlflow.log_metric("threshold", threshold)
        mlflow.log_metrics(results)
        print(f"threshold {threshold:.4f}")
        print_results(results)

    suffix = "_local" if local_only else ""
    np.save(f"data/elliptic/xgb_scores{suffix}.npy", scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only", action="store_true")
    main(parser.parse_args().local_only)