import numpy as np
import pandas as pd
import torch

DATA_DIR = "data/elliptic"
TRAIN_LAST_STEP = 29  # steps 1-29 train
VALID_LAST_STEP = 34  # steps 30-34 validation (early stopping, threshold); 35-49 test

LABELS = {"1": 1, "2": 0, "unknown": -1}  # 1 = illicit, 0 = licit, -1 = unlabeled


def load_elliptic(data_dir: str = DATA_DIR, local_only: bool = False) -> dict:
    features = pd.read_csv(f"{data_dir}/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(f"{data_dir}/elliptic_txs_classes.csv")
    edges = pd.read_csv(f"{data_dir}/elliptic_txs_edgelist.csv")

    # Column 0 is the transaction id, column 1 the time step, the rest are features.
    features = features.rename(columns={0: "txId", 1: "time_step"})
    classes.columns = ["txId", "class"]
    df = features.merge(classes, on="txId", how="left")
    df["label"] = df["class"].astype(str).map(LABELS)

    # Graph nodes are numbered 0..n-1 in the order of this table.
    node_index = {tx: i for i, tx in enumerate(df["txId"])}
    edges = edges.rename(columns={edges.columns[0]: "src", edges.columns[1]: "dst"})
    edges["src"] = edges["src"].map(node_index)
    edges["dst"] = edges["dst"].map(node_index)
    edges = edges.dropna().astype("int64")
    edge_index = torch.tensor(edges[["src", "dst"]].values.T, dtype=torch.long)

    feature_cols = [c for c in df.columns if isinstance(c, int)]
    x = df[feature_cols].values.astype(np.float32)
    if local_only:
        x = x[:, :93]  # drop the 72 pre-aggregated neighbour features

    time_step = df["time_step"].values
    y = df["label"].values

    labeled = y != -1
    return {
        "x": x,
        "y": y,
        "time_step": time_step,
        "edge_index": edge_index,
        "train_mask": labeled & (time_step <= TRAIN_LAST_STEP),
        "valid_mask": labeled & (time_step > TRAIN_LAST_STEP) & (time_step <= VALID_LAST_STEP),
        "test_mask": labeled & (time_step > VALID_LAST_STEP),
    }


if __name__ == "__main__":
    data = load_elliptic()
    y, steps = data["y"], data["time_step"]
    print("nodes:", len(y), "| edges:", data["edge_index"].shape[1], "| features:", data["x"].shape[1])
    print("labeled:", int((y != -1).sum()), "| illicit:", int((y == 1).sum()), "| licit:", int((y == 0).sum()))
    for name in ["train_mask", "valid_mask", "test_mask"]:
        mask = data[name]
        print(f"{name}: {int(mask.sum()):,} nodes, illicit rate {y[mask].mean():.3%}")
    print("local-only features:", load_elliptic(local_only=True)["x"].shape[1])
