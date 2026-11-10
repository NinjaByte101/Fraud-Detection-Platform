import mlflow
import numpy as np
from sklearn.metrics import average_precision_score

from src.graph.load_elliptic import load_elliptic
from src.graph.metrics import best_f1_threshold, evaluate, print_results


def main():
    data = load_elliptic()
    y, steps = data["y"], data["time_step"]
    valid, test = data["valid_mask"], data["test_mask"]

    xgb_scores = np.load("data/elliptic/xgb_scores.npy")
    gnn_scores = np.load("data/elliptic/gnn_scores.npy")

    # weight = share of the GNN score (0 = XGBoost only, 1 = GNN only)
    best_weight, best_pr = 0.0, -1.0
    for weight in np.linspace(0, 1, 21):
        blended = (1 - weight) * xgb_scores + weight * gnn_scores
        pr = average_precision_score(y[valid], blended[valid])
        if pr > best_pr:
            best_weight, best_pr = float(weight), pr
    print(f"best GNN weight on validation: {best_weight:.2f} (valid PR-AUC {best_pr:.3f})")

    blended = (1 - best_weight) * xgb_scores + best_weight * gnn_scores
    candidates = {"xgboost": xgb_scores, "graphsage": gnn_scores, "blend": blended}

    mlflow.set_experiment("fraud-elliptic")
    with mlflow.start_run(run_name="blend"):
        mlflow.log_param("gnn_weight", best_weight)
        for name, scores in candidates.items():
            threshold = best_f1_threshold(y[valid], scores[valid])
            results = evaluate(y, scores, steps, test, threshold)
            mlflow.log_metrics({f"{name}_{k}": v for k, v in results.items()})
            print(f"\n{name} (threshold {threshold:.4f})")
            print_results(results)

    np.save("data/elliptic/blend_scores.npy", blended)


if __name__ == "__main__":
    main()