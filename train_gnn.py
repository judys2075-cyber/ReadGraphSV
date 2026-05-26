#!/usr/bin/env python3
"""Train a baseline GraphSAGE classifier on ReadGraphSV candidate graphs."""

import argparse
import csv
import logging
import os
import random

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from utils_graph import FEATURE_NAMES, ReadGraphSAGE


METRIC_FIELDS = ["epoch", "train_loss", "precision", "recall", "f1", "auc"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Input graphs/dataset.pt")
    parser.add_argument("--model_out", required=True, help="Output model checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden channel count")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sanitize_graph(graph):
    """Drop metadata before batching so PyG only collates tensor fields."""
    y = graph.y.view(-1).float() if hasattr(graph, "y") else torch.tensor([0.0])
    return Data(x=graph.x.float(), edge_index=graph.edge_index.long(), y=y)


def split_dataset(graphs, test_ratio, seed):
    n = len(graphs)
    if n <= 1:
        return graphs, graphs

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    test_size = max(1, int(round(n * test_ratio)))
    test_size = min(test_size, n - 1)
    test_indices = set(indices[:test_size])
    train = [graphs[i] for i in range(n) if i not in test_indices]
    test = [graphs[i] for i in range(n) if i in test_indices]
    return train, test


def compute_metrics(labels, probs, threshold=0.5):
    if len(labels) == 0:
        return 0.0, 0.0, 0.0, float("nan")
    preds = (np.asarray(probs) >= threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    if len(set(labels.tolist())) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels, probs))
    return float(precision), float(recall), float(f1), auc


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    labels = []
    probs = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        prob = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        label = batch.y.view(-1).detach().cpu().numpy().tolist()
        probs.extend(prob)
        labels.extend(label)
    return labels, probs


def write_empty_metrics(path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ensure_parent_dir(args.model_out)
    project_root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "train_metrics.csv")

    if not os.path.exists(args.dataset):
        logging.warning("Dataset not found: %s", args.dataset)
        write_empty_metrics(metrics_path)
        return

    raw_graphs = safe_torch_load(args.dataset)
    graphs = [sanitize_graph(graph) for graph in raw_graphs if hasattr(graph, "x") and graph.x.numel() > 0]
    if not graphs:
        logging.warning("Dataset has no usable graphs: %s", args.dataset)
        write_empty_metrics(metrics_path)
        return

    train_graphs, test_graphs = split_dataset(graphs, args.test_ratio, args.seed)
    in_channels = int(train_graphs[0].x.size(1))
    train_loader = DataLoader(train_graphs, batch_size=min(32, len(train_graphs)), shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=min(64, len(test_graphs)), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReadGraphSAGE(in_channels=in_channels, hidden_channels=args.hidden).to(device)

    train_labels = np.array([int(graph.y.item()) for graph in train_graphs], dtype=int)
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    if positives > 0 and negatives > 0:
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)
        logging.warning("Training split has only one class; using pos_weight=1.0")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    with open(metrics_path, "w", newline="") as metrics_handle:
        writer = csv.DictWriter(metrics_handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_graphs = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index, batch.batch)
                labels = batch.y.view(-1).float()
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * int(labels.numel())
                total_graphs += int(labels.numel())

            train_loss = total_loss / max(1, total_graphs)
            labels, probs = predict_loader(model, test_loader, device)
            precision, recall, f1, auc = compute_metrics(labels, probs)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "precision": round(precision, 6),
                    "recall": round(recall, 6),
                    "f1": round(f1, 6),
                    "auc": "" if np.isnan(auc) else round(auc, 6),
                }
            )
            metrics_handle.flush()

            logging.info(
                "epoch=%d loss=%.4f precision=%.3f recall=%.3f f1=%.3f auc=%s",
                epoch,
                train_loss,
                precision,
                recall,
                f1,
                "NA" if np.isnan(auc) else f"{auc:.3f}",
            )

            if f1 > best_f1:
                best_f1 = f1
                checkpoint = {
                    "model_state": model.state_dict(),
                    "in_channels": in_channels,
                    "hidden_channels": args.hidden,
                    "feature_names": FEATURE_NAMES,
                    "best_f1": best_f1,
                    "epoch": epoch,
                }
                torch.save(checkpoint, args.model_out)

    logging.info("Saved best model to %s with best_f1=%.4f", args.model_out, best_f1)
    logging.info("Wrote training metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
