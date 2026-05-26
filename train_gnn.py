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


METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "threshold",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_auc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_auc",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Input graphs/dataset.pt")
    parser.add_argument("--model_out", required=True, help="Output model checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden channel count")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Held-out test split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed probability threshold when --auto_threshold is off")
    parser.add_argument(
        "--auto_threshold",
        action="store_true",
        help="Search the best validation-set threshold for F1 at each epoch",
    )
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


def split_dataset(graphs, val_ratio, test_ratio, seed):
    """Split graphs into train/validation/test sets with small-dataset fallbacks."""
    n_graphs = len(graphs)
    if n_graphs <= 1:
        return graphs, graphs, graphs

    indices = list(range(n_graphs))
    rng = random.Random(seed)
    rng.shuffle(indices)

    test_size = max(1, int(round(n_graphs * test_ratio))) if test_ratio > 0 else 0
    val_size = max(1, int(round(n_graphs * val_ratio))) if val_ratio > 0 else 0

    while test_size + val_size >= n_graphs:
        if val_size > 0:
            val_size -= 1
        elif test_size > 0:
            test_size -= 1
        else:
            break

    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size :]

    train = [graphs[i] for i in train_indices] or [graphs[indices[-1]]]
    val = [graphs[i] for i in val_indices] or train
    test = [graphs[i] for i in test_indices] or val
    return train, val, test


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


def select_best_threshold(labels, probs, default_threshold=0.5):
    """Choose the validation threshold with maximum F1."""
    if len(labels) == 0:
        return float(default_threshold)

    best_threshold = float(default_threshold)
    best_key = (-1.0, -1.0, -1.0)
    for threshold in np.arange(0.05, 1.0, 0.05):
        precision, recall, f1, _auc = compute_metrics(labels, probs, threshold=float(threshold))
        key = (f1, recall, precision)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


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


def format_metric(value):
    if isinstance(value, float) and np.isnan(value):
        return ""
    return round(float(value), 6)


def write_test_summary(path, metrics, threshold, epoch):
    precision, recall, f1, auc = metrics
    with open(path, "w") as handle:
        print(f"Best epoch: {epoch}", file=handle)
        print(f"Selected threshold: {threshold:.6f}", file=handle)
        print(f"Test Precision: {precision:.6f}", file=handle)
        print(f"Test Recall: {recall:.6f}", file=handle)
        print(f"Test F1: {f1:.6f}", file=handle)
        print("Test AUC: NA" if np.isnan(auc) else f"Test AUC: {auc:.6f}", file=handle)


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
    test_summary_path = os.path.join(results_dir, "test_metrics.txt")

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

    train_graphs, val_graphs, test_graphs = split_dataset(graphs, args.val_ratio, args.test_ratio, args.seed)
    in_channels = int(train_graphs[0].x.size(1))
    train_loader = DataLoader(train_graphs, batch_size=min(32, len(train_graphs)), shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=min(64, len(val_graphs)), shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=min(64, len(test_graphs)), shuffle=False)

    logging.info(
        "Dataset split: train=%d val=%d test=%d",
        len(train_graphs),
        len(val_graphs),
        len(test_graphs),
    )

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

    best_val_f1 = -1.0
    best_epoch = 0
    best_threshold = float(args.threshold)
    best_checkpoint = None

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

            val_labels, val_probs = predict_loader(model, val_loader, device)
            threshold = (
                select_best_threshold(val_labels, val_probs, args.threshold)
                if args.auto_threshold
                else float(args.threshold)
            )
            val_precision, val_recall, val_f1, val_auc = compute_metrics(val_labels, val_probs, threshold)

            test_labels, test_probs = predict_loader(model, test_loader, device)
            test_precision, test_recall, test_f1, test_auc = compute_metrics(test_labels, test_probs, threshold)

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "threshold": round(threshold, 6),
                    "val_precision": format_metric(val_precision),
                    "val_recall": format_metric(val_recall),
                    "val_f1": format_metric(val_f1),
                    "val_auc": format_metric(val_auc),
                    "test_precision": format_metric(test_precision),
                    "test_recall": format_metric(test_recall),
                    "test_f1": format_metric(test_f1),
                    "test_auc": format_metric(test_auc),
                }
            )
            metrics_handle.flush()

            logging.info(
                "epoch=%d loss=%.4f threshold=%.2f val_f1=%.3f test_f1=%.3f test_auc=%s",
                epoch,
                train_loss,
                threshold,
                val_f1,
                test_f1,
                "NA" if np.isnan(test_auc) else f"{test_auc:.3f}",
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_threshold = threshold
                best_checkpoint = {
                    "model_state": {
                        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                    },
                    "in_channels": in_channels,
                    "hidden_channels": args.hidden,
                    "feature_names": FEATURE_NAMES,
                    "best_val_f1": best_val_f1,
                    "best_threshold": best_threshold,
                    "epoch": best_epoch,
                    "auto_threshold": bool(args.auto_threshold),
                }
                torch.save(best_checkpoint, args.model_out)

    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint["model_state"])
        test_labels, test_probs = predict_loader(model, test_loader, device)
        final_test_metrics = compute_metrics(test_labels, test_probs, best_threshold)
        write_test_summary(test_summary_path, final_test_metrics, best_threshold, best_epoch)
        precision, recall, f1, auc = final_test_metrics
        logging.info(
            "Final test metrics at threshold %.3f: precision=%.3f recall=%.3f f1=%.3f auc=%s",
            best_threshold,
            precision,
            recall,
            f1,
            "NA" if np.isnan(auc) else f"{auc:.3f}",
        )

    logging.info("Saved best validation-F1 model to %s", args.model_out)
    logging.info("Wrote training metrics to %s", metrics_path)
    logging.info("Wrote final test metrics to %s", test_summary_path)


if __name__ == "__main__":
    main()
