import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def best_f1_threshold(y_true, scores):
    """Pick the score cutoff with the best F1 on the validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    return float(thresholds[np.argmax(f1)])


def summarize(y, scores, mask, threshold):
    y_true, s = y[mask], scores[mask]
    predicted = s >= threshold
    tp = int((predicted & (y_true == 1)).sum())
    precision = tp / max(int(predicted.sum()), 1)
    recall = tp / max(int((y_true == 1).sum()), 1)
    return {
        "pr_auc": float(average_precision_score(y_true, s)),
        "base_rate": float(y_true.mean()),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + 1e-9),
    }


def evaluate(y, scores, steps, test_mask, threshold):
    """Test metrics overall, before the step-43 shutdown, and after it."""
    groups = {
        "test": test_mask,
        "test_35_42": test_mask & (steps <= 42),
        "test_43_49": test_mask & (steps >= 43),
    }
    results = {}
    for group, mask in groups.items():
        for metric, value in summarize(y, scores, mask, threshold).items():
            results[f"{group}_{metric}"] = value
    return results


def print_results(results):
    for group in ["test", "test_35_42", "test_43_49"]:
        print(
            f"{group:11s} PR-AUC {results[f'{group}_pr_auc']:.3f} "
            f"(base rate {results[f'{group}_base_rate']:.3f}) | "
            f"precision {results[f'{group}_precision']:.3f} | "
            f"recall {results[f'{group}_recall']:.3f} | F1 {results[f'{group}_f1']:.3f}"
        )