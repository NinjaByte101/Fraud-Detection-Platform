import argparse

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.features.build_features import (
    FEATURE_SETS,
    add_features,
    load_risky_transactions,
    time_split,
)

DATA_PATH = "data/paysim.csv"


def best_f1_threshold(y_true, scores):
    """Pick the score cutoff with the best F1 on the validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    return thresholds[np.argmax(f1)]


def main(feature_set: str):
    cols = FEATURE_SETS[feature_set]
    df = add_features(load_risky_transactions(DATA_PATH))
    train, valid, test = time_split(df)

    X_train, y_train = train[cols], train["isFraud"]
    X_valid, y_valid = valid[cols], valid["isFraud"]
    X_test, y_test = test[cols], test["isFraud"]

    # Fraud is rare, so each fraud row gets a much larger weight.
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": pos_weight,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 30,
        "tree_method": "hist",
        "random_state": 42,
    }

    mlflow.set_experiment("fraud-xgboost")
    with mlflow.start_run(run_name=f"xgb-{feature_set}"):
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_params({k: round(float(v), 2) if k == "scale_pos_weight" else v
                           for k, v in params.items()})

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=50)

        valid_scores = model.predict_proba(X_valid)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        threshold = best_f1_threshold(y_valid, valid_scores)

        test_pred = test_scores >= threshold
        tp = int(((test_pred == 1) & (y_test == 1)).sum())
        precision = tp / max(int(test_pred.sum()), 1)
        recall = tp / int(y_test.sum())

        metrics = {
            "valid_pr_auc": average_precision_score(y_valid, valid_scores),
            "test_pr_auc": average_precision_score(y_test, test_scores),
            "test_base_rate": float(y_test.mean()),
            "threshold": float(threshold),
            "test_precision": precision,
            "test_recall": recall,
        }
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, name="model")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="serving")
    args = parser.parse_args()
    main(args.features)