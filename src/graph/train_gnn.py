import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

from src.graph.load_elliptic import load_elliptic
from src.graph.metrics import best_f1_threshold, evaluate, print_results


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.out = torch.nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)


def main(epochs=200, lr=0.005, patience=30):
    torch.manual_seed(42)
    data = load_elliptic()
    y_np, steps = data["y"], data["time_step"]
    train, valid, test = data["train_mask"], data["valid_mask"], data["test_mask"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(data["x"]).to(device)
    # Money flows one way, but suspicious neighbours matter in both directions.
    edge_index = to_undirected(data["edge_index"]).to(device)
    # Unlabeled nodes are marked -1; they never enter the loss, so 0 is a safe filler.
    y = torch.tensor(np.where(y_np == -1, 0, y_np)).long().to(device)
    train_idx = torch.tensor(np.where(train)[0], device=device)

    pos_weight = float((y_np[train] == 0).sum() / (y_np[train] == 1).sum())
    class_weight = torch.tensor([1.0, pos_weight], device=device)

    model = GraphSAGE(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    mlflow.set_experiment("fraud-elliptic")
    with mlflow.start_run(run_name="graphsage"):
        mlflow.log_params({"hidden": 128, "dropout": 0.3, "lr": lr, "epochs": epochs,
                           "pos_weight": round(pos_weight, 2), "device": device})

        best_pr, best_epoch, best_scores = -1.0, 0, None
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x, edge_index)[train_idx], y[train_idx], weight=class_weight)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                scores = torch.softmax(model(x, edge_index), dim=1)[:, 1].cpu().numpy()
            valid_pr = average_precision_score(y_np[valid], scores[valid])
            mlflow.log_metrics({"train_loss": float(loss), "valid_pr_auc": valid_pr}, step=epoch)

            if valid_pr > best_pr:
                best_pr, best_epoch, best_scores = valid_pr, epoch, scores
            if epoch % 10 == 0:
                print(f"epoch {epoch:3d} | loss {loss:.4f} | valid PR-AUC {valid_pr:.3f}")
            if epoch - best_epoch >= patience:
                print(f"early stop at epoch {epoch}, best was {best_epoch}")
                break

        threshold = best_f1_threshold(y_np[valid], best_scores[valid])
        results = evaluate(y_np, best_scores, steps, test, threshold)
        mlflow.log_metrics({"best_epoch": best_epoch, "threshold": threshold, **results})
        print(f"best epoch {best_epoch} | threshold {threshold:.4f}")
        print_results(results)

    np.save("data/elliptic/gnn_scores.npy", best_scores)


if __name__ == "__main__":
    main()