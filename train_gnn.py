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


def parse_args(argv=None):
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
    parser.add_argument("--init_model", default=None, help="Optional checkpoint used to initialize fine-tuning")
    parser.add_argument("--val_dataset", default=None, help="Optional independent validation graph dataset")
    parser.add_argument("--test_dataset", default=None, help="Optional independent test graph dataset")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience measured in epochs")
    parser.add_argument(
        "--class_weight",
        choices=["auto", "none"],
        default="auto",
        help="Class weighting mode for BCEWithLogitsLoss",
    )
    parser.add_argument("--freeze_encoder", action="store_true", help="Only train the final classifier layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


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


def infer_feature_names(raw_graphs, in_channels):
    for graph in raw_graphs:
        feature_names = getattr(graph, "feature_names", None)
        if feature_names and len(feature_names) == in_channels:
            return list(feature_names)
    if len(FEATURE_NAMES) == in_channels:
        return FEATURE_NAMES
    return [f"feature_{idx}" for idx in range(in_channels)]


def load_graph_dataset(path, required=False):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Dataset not found: {path}")
        logging.warning("Dataset not found: %s", path)
        return [], []

    raw_graphs = safe_torch_load(path)
    graphs = [sanitize_graph(graph) for graph in raw_graphs if hasattr(graph, "x") and graph.x.numel() > 0]
    if not graphs:
        message = f"Dataset has no usable graphs: {path}"
        if required:
            raise ValueError(message)
        logging.warning(message)
    return raw_graphs, graphs


def validate_feature_dimensions(name, graphs, expected):
    for idx, graph in enumerate(graphs):
        observed = int(graph.x.size(1))
        if observed != expected:
            raise ValueError(
                f"{name} graph {idx} has {observed} features, expected {expected}; "
                "all train/validation/test datasets must use the same graph feature schema"
            )


def clone_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_initial_model(model, init_model, expected_in_channels, expected_hidden):
    checkpoint = safe_torch_load(init_model, map_location="cpu")
    if "model_state" not in checkpoint:
        raise ValueError(f"Initial checkpoint has no model_state: {init_model}")

    init_in_channels = int(checkpoint.get("in_channels", expected_in_channels))
    init_hidden = int(checkpoint.get("hidden_channels", expected_hidden))
    if init_in_channels != expected_in_channels:
        raise ValueError(
            f"Initial model expects {init_in_channels} input features, "
            f"but training dataset has {expected_in_channels}"
        )
    if init_hidden != expected_hidden:
        raise ValueError(
            f"Initial model hidden size is {init_hidden}, but --hidden is {expected_hidden}; "
            "use the same hidden size as the checkpoint"
        )

    model.load_state_dict(checkpoint["model_state"], strict=True)
    logging.info("Loaded initial model weights from %s", init_model)
    return checkpoint


def freeze_encoder_parameters(model):
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if name.startswith("classifier."):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    logging.info("Freeze encoder enabled: frozen=%d trainable=%d parameters", frozen, trainable)


def trainable_parameters(model):
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters remain; check --freeze_encoder and model architecture")
    return params


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


def split_train_val(graphs, val_ratio, seed):
    """Split graphs into train/validation sets when an external test set is used."""
    train, val, _test = split_dataset(graphs, val_ratio=val_ratio, test_ratio=0.0, seed=seed)
    return train, val


def resolve_datasets(args, graphs):
    """Resolve train/validation/test graphs while preserving legacy random splitting."""
    if args.val_dataset:
        _raw_val, val_graphs = load_graph_dataset(args.val_dataset, required=True)
        train_graphs = graphs
        if args.test_dataset:
            _raw_test, test_graphs = load_graph_dataset(args.test_dataset, required=True)
        else:
            test_graphs = []
        split_mode = "external_validation"
    elif args.test_dataset:
        train_graphs, val_graphs = split_train_val(graphs, args.val_ratio, args.seed)
        _raw_test, test_graphs = load_graph_dataset(args.test_dataset, required=True)
        split_mode = "random_train_val_external_test"
    else:
        train_graphs, val_graphs, test_graphs = split_dataset(graphs, args.val_ratio, args.test_ratio, args.seed)
        split_mode = "random_train_val_test"

    return train_graphs, val_graphs, test_graphs, split_mode


def build_loader(graphs, batch_size, shuffle=False):
    if not graphs:
        return None
    return DataLoader(graphs, batch_size=min(batch_size, len(graphs)), shuffle=shuffle)


def resolve_pos_weight(train_graphs, mode, device):
    if mode == "none":
        logging.info("Class weighting disabled")
        return None

    train_labels = np.array([int(graph.y.item()) for graph in train_graphs], dtype=int)
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    if positives > 0 and negatives > 0:
        value = negatives / positives
        logging.info("Class weighting auto: positives=%d negatives=%d pos_weight=%.4f", positives, negatives, value)
        return torch.tensor([value], dtype=torch.float32, device=device)

    logging.warning("Training split has only one class; using pos_weight=1.0")
    return torch.tensor([1.0], dtype=torch.float32, device=device)


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
    if loader is None:
        return [], []
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


def write_no_test_summary(path, threshold, epoch):
    with open(path, "w") as handle:
        print(f"Best epoch: {epoch}", file=handle)
        print(f"Selected threshold: {threshold:.6f}", file=handle)
        print("No independent test dataset was evaluated.", file=handle)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.init_model and os.path.abspath(args.init_model) == os.path.abspath(args.model_out):
        raise ValueError("--model_out must be different from --init_model when fine-tuning")

    ensure_parent_dir(args.model_out)
    project_root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "train_metrics.csv")
    test_summary_path = os.path.join(results_dir, "test_metrics.txt")

    raw_graphs, graphs = load_graph_dataset(args.dataset)
    if not graphs:
        logging.warning("Dataset has no usable graphs: %s", args.dataset)
        write_empty_metrics(metrics_path)
        return

    train_graphs, val_graphs, test_graphs, split_mode = resolve_datasets(args, graphs)
    in_channels = int(train_graphs[0].x.size(1))
    feature_names = infer_feature_names(raw_graphs, in_channels)
    validate_feature_dimensions("train", train_graphs, in_channels)
    validate_feature_dimensions("validation", val_graphs, in_channels)
    validate_feature_dimensions("test", test_graphs, in_channels)

    train_loader = build_loader(train_graphs, batch_size=32, shuffle=True)
    val_loader = build_loader(val_graphs, batch_size=64, shuffle=False)
    test_loader = build_loader(test_graphs, batch_size=64, shuffle=False)

    logging.info(
        "Dataset split mode=%s: train=%d val=%d test=%d",
        split_mode,
        len(train_graphs),
        len(val_graphs),
        len(test_graphs),
    )
    if args.val_dataset:
        logging.info("Using independent validation dataset: %s", args.val_dataset)
    if args.test_dataset:
        logging.info("Using independent test dataset: %s", args.test_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReadGraphSAGE(in_channels=in_channels, hidden_channels=args.hidden).to(device)

    init_checkpoint = None
    if args.init_model:
        init_checkpoint = load_initial_model(model, args.init_model, in_channels, args.hidden)
        init_feature_names = init_checkpoint.get("feature_names")
        if init_feature_names and list(init_feature_names) != list(feature_names):
            logging.warning("Initial model feature names differ from the training dataset feature names")

    if args.freeze_encoder:
        freeze_encoder_parameters(model)

    pos_weight = resolve_pos_weight(train_graphs, args.class_weight, device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=args.lr)

    best_val_f1 = -1.0
    best_epoch = 0
    best_threshold = float(args.threshold)
    best_checkpoint = None
    epochs_without_improvement = 0
    early_stop_enabled = args.patience is not None and args.patience > 0

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
            if test_loader is None:
                test_precision = test_recall = test_f1 = test_auc = float("nan")
            else:
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
                "epoch=%d loss=%.4f threshold=%.2f val_precision=%.3f val_recall=%.3f val_f1=%.3f val_auc=%s",
                epoch,
                train_loss,
                threshold,
                val_precision,
                val_recall,
                val_f1,
                "NA" if np.isnan(val_auc) else f"{val_auc:.3f}",
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_threshold = threshold
                best_checkpoint = {
                    "model_state": clone_state_dict(model),
                    "in_channels": in_channels,
                    "hidden_channels": args.hidden,
                    "feature_names": feature_names,
                    "best_val_f1": best_val_f1,
                    "best_threshold": best_threshold,
                    "epoch": best_epoch,
                    "auto_threshold": bool(args.auto_threshold),
                    "init_model": args.init_model or "",
                    "val_dataset": args.val_dataset or "",
                    "test_dataset": args.test_dataset or "",
                    "split_mode": split_mode,
                    "class_weight": args.class_weight,
                    "freeze_encoder": bool(args.freeze_encoder),
                    "patience": int(args.patience),
                }
                torch.save(best_checkpoint, args.model_out)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if early_stop_enabled and epochs_without_improvement >= args.patience:
                    logging.info(
                        "Early stopping at epoch %d after %d epochs without validation F1 improvement",
                        epoch,
                        epochs_without_improvement,
                    )
                    break

    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint["model_state"])
        if test_loader is not None:
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
        else:
            write_no_test_summary(test_summary_path, best_threshold, best_epoch)
            logging.info("No independent test dataset was evaluated")

    logging.info("Saved best validation-F1 model to %s", args.model_out)
    logging.info("Wrote training metrics to %s", metrics_path)
    logging.info("Wrote final test metrics to %s", test_summary_path)


if __name__ == "__main__":
    main()
