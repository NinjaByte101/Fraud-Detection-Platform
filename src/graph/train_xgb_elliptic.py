import mlflow
import numpy as np
import xgboost as xgb

from src.graph.load_elliptic import load_elliptic
from src.graph.metrics import best_f1_threshold, evaluate, print_results


def main():
    data = load_elliptic()
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

    mlflow.set_experiment("fraud-elliptic")
    with mlflow.start_run(run_name="xgb-node-features"):
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(x[train], y[train], eval_set=[(x[valid], y[valid])], verbose=50)

        scores = model.predict_proba(x)[:, 1]
        threshold = best_f1_threshold(y[valid], scores[valid])
        results = evaluate(y, scores, steps, test, threshold)

        mlflow.log_metric("threshold", threshold)
        mlflow.log_metrics(results)
        print(f"threshold {threshold:.4f}")
        print_results(results)

    np.save("data/elliptic/xgb_scores.npy", scores)


if __name__ == "__main__":
    main()