#!/usr/bin/env python3
"""Run a trained ReadGraphSV GNN model on candidate graphs."""

import argparse
import csv
import logging
import os

import torch

from utils_graph import ReadGraphSAGE


FIELDS = [
    "candidate_id",
    "chrom",
    "start",
    "end",
    "svtype",
    "median_svlen",
    "support_read_count",
    "label",
    "gnn_prob",
    "gnn_pred",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Input graphs/dataset.pt")
    parser.add_argument("--model", required=True, help="Model checkpoint")
    parser.add_argument("--out", required=True, help="Output predictions.tsv")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
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


def empty_output(path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()


def graph_metadata(graph):
    meta = getattr(graph, "metadata", {}) or {}
    return {
        "candidate_id": str(meta.get("candidate_id", "")),
        "chrom": str(meta.get("chrom", "")),
        "start": int(meta.get("start", 0)),
        "end": int(meta.get("end", 0)),
        "svtype": str(meta.get("svtype", "")),
        "median_svlen": int(meta.get("median_svlen", 0)),
        "support_read_count": int(meta.get("support_read_count", 0)),
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_parent_dir(args.out)

    if not os.path.exists(args.dataset):
        logging.warning("Dataset not found: %s", args.dataset)
        empty_output(args.out)
        return
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"model checkpoint not found: {args.model}")

    graphs = safe_torch_load(args.dataset)
    if not graphs:
        logging.warning("Dataset has no graphs: %s", args.dataset)
        empty_output(args.out)
        return

    checkpoint = safe_torch_load(args.model)
    in_channels = int(checkpoint.get("in_channels", graphs[0].x.size(1)))
    hidden_channels = int(checkpoint.get("hidden_channels", 64))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReadGraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows = []
    with torch.no_grad():
        for graph in graphs:
            if not hasattr(graph, "x") or graph.x.numel() == 0:
                continue
            x = graph.x.float().to(device)
            edge_index = graph.edge_index.long().to(device)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
            prob = float(torch.sigmoid(model(x, edge_index, batch)).item())
            pred = 1 if prob >= args.threshold else 0
            label = int(graph.y.view(-1)[0].item()) if hasattr(graph, "y") else -1
            meta = graph_metadata(graph)
            rows.append(
                {
                    **meta,
                    "label": label,
                    "gnn_prob": round(prob, 6),
                    "gnn_pred": pred,
                }
            )

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logging.info("Wrote %d predictions to %s", len(rows), args.out)


if __name__ == "__main__":
    main()
