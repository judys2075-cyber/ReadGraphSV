#!/usr/bin/env python3
"""Evaluate raw candidates and GNN-filtered candidates."""

import argparse
import logging
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="Input predictions.tsv")
    parser.add_argument("--threshold", type=float, default=0.5, help="GNN probability threshold")
    parser.add_argument("--out", required=True, help="Output evaluation.txt")
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def confusion(labels, preds):
    labels = [int(x) for x in labels]
    preds = [int(x) for x in preds]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def format_block(name, tp, fp, fn, tn):
    precision, recall, f1 = prf(tp, fp, fn)
    return [
        f"{name} Precision: {precision:.6f}",
        f"{name} Recall: {recall:.6f}",
        f"{name} F1: {f1:.6f}",
        f"{name} TP: {tp}",
        f"{name} FP: {fp}",
        f"{name} FN: {fn}",
        f"{name} TN: {tn}",
    ]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_parent_dir(args.out)

    if not os.path.exists(args.pred):
        logging.warning("Prediction file not found: %s", args.pred)
        with open(args.out, "w") as handle:
            handle.write("No predictions found.\n")
        return

    try:
        pred = pd.read_csv(args.pred, sep="\t")
    except pd.errors.EmptyDataError:
        pred = pd.DataFrame()
    if pred.empty or "label" not in pred.columns:
        logging.warning("Prediction file is empty or has no label column: %s", args.pred)
        with open(args.out, "w") as handle:
            handle.write("No labeled predictions to evaluate.\n")
        return

    pred = pred[pred["label"].isin([0, 1, "0", "1"])].copy()
    if pred.empty:
        with open(args.out, "w") as handle:
            handle.write("No labels with values 0/1 to evaluate.\n")
        return

    labels = pred["label"].astype(int).tolist()
    raw_preds = [1] * len(labels)
    gnn_preds = (pd.to_numeric(pred["gnn_prob"], errors="coerce").fillna(0.0) >= args.threshold).astype(int).tolist()

    raw_tp, raw_fp, raw_fn, raw_tn = confusion(labels, raw_preds)
    gnn_tp, gnn_fp, gnn_fn, gnn_tn = confusion(labels, gnn_preds)

    fp_reduction = raw_fp - gnn_fp
    fp_reduction_rate = fp_reduction / raw_fp if raw_fp > 0 else 0.0
    tp_loss = raw_tp - gnn_tp
    tp_loss_rate = tp_loss / raw_tp if raw_tp > 0 else 0.0

    lines = []
    lines.extend(format_block("Raw", raw_tp, raw_fp, raw_fn, raw_tn))
    lines.append("")
    lines.extend(format_block("GNN", gnn_tp, gnn_fp, gnn_fn, gnn_tn))
    lines.append("")
    lines.append(f"FP reduction: {fp_reduction}")
    lines.append(f"FP reduction rate: {fp_reduction_rate:.6f}")
    lines.append(f"TP loss: {tp_loss}")
    lines.append(f"TP loss rate: {tp_loss_rate:.6f}")

    with open(args.out, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    logging.info("Wrote evaluation to %s", args.out)


if __name__ == "__main__":
    main()
