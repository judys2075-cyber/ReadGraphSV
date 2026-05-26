#!/usr/bin/env python3
"""Build region-level PyTorch Geometric graphs from candidates and signals."""

import argparse
import logging
import os

import pandas as pd
import torch
from torch_geometric.data import Data

from utils_graph import (
    FEATURE_NAMES,
    bool_numeric,
    log1p_norm,
    numeric,
    strand_numeric,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", required=True, help="Input signals.tsv")
    parser.add_argument("--candidates", required=True, help="Input candidates_labeled.tsv")
    parser.add_argument("--out", required=True, help="Output graphs/dataset.pt")
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def svtype_flags(svtype):
    return 1.0 if svtype == "DEL" else 0.0, 1.0 if svtype == "INS" else 0.0


def parse_signal_indices(value):
    if pd.isna(value):
        return []
    indices = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            indices.append(int(item))
        except ValueError:
            continue
    return indices


def candidate_feature(row):
    sv_del, sv_ins = svtype_flags(str(row["svtype"]))
    return [
        sv_del,
        sv_ins,
        log1p_norm(row.get("median_svlen", 0)),
        log1p_norm(row.get("support_read_count", 0), scale=100.0),
        min(1.0, numeric(row.get("mean_mapq", 0)) / 60.0),
        min(1.0, numeric(row.get("std_pos", 0)) / 1000.0),
        min(1.0, numeric(row.get("std_svlen", 0)) / max(1.0, abs(numeric(row.get("median_svlen", 0))))),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]


def read_feature(signal, candidate_center, candidate_svlen):
    svtype = str(signal["svtype"])
    sv_del, sv_ins = svtype_flags(svtype)
    svlen = abs(numeric(signal.get("svlen", 0)))
    candidate_svlen = max(1.0, abs(float(candidate_svlen)))
    return [
        sv_del,
        sv_ins,
        log1p_norm(svlen),
        0.0,
        0.0,
        0.0,
        0.0,
        log1p_norm(signal.get("read_len", 0), scale=100000.0),
        min(1.0, numeric(signal.get("mapq", 0)) / 60.0),
        strand_numeric(signal.get("strand", "")),
        min(1.0, abs(numeric(signal.get("event_pos", 0)) - candidate_center) / 1000.0),
        min(1.0, abs(svlen - candidate_svlen) / candidate_svlen),
        bool_numeric(signal.get("has_sa", 0)),
        bool_numeric(signal.get("is_supplementary", 0)),
        0.0,
        1.0,
    ]


def signals_for_candidate(signals, row):
    indices = parse_signal_indices(row.get("signal_indices", ""))
    if indices:
        return signals.loc[signals.index.intersection(indices)].copy()

    chrom = str(row["chrom"])
    svtype = str(row["svtype"])
    start = int(row["start"])
    end = int(row["end"])
    margin = 500
    return signals[
        (signals["chrom"].astype(str) == chrom)
        & (signals["svtype"].astype(str) == svtype)
        & (signals["event_pos"].between(start - margin, end + margin))
    ].copy()


def build_one_graph(signals, row):
    candidate_center = numeric(row.get("start", 0))
    candidate_svlen = max(1.0, abs(numeric(row.get("median_svlen", 0))))
    evidence = signals_for_candidate(signals, row)

    node_features = [candidate_feature(row)]
    for _, signal in evidence.iterrows():
        node_features.append(read_feature(signal, candidate_center, candidate_svlen))

    edge_src = []
    edge_dst = []
    read_node_count = len(node_features) - 1

    for node_idx in range(1, read_node_count + 1):
        edge_src.extend([0, node_idx])
        edge_dst.extend([node_idx, 0])

    evidence_records = evidence.reset_index(drop=True).to_dict("records")
    for i in range(len(evidence_records)):
        for j in range(i + 1, len(evidence_records)):
            pos_i = numeric(evidence_records[i].get("event_pos", 0))
            pos_j = numeric(evidence_records[j].get("event_pos", 0))
            len_i = abs(numeric(evidence_records[i].get("svlen", 0)))
            len_j = abs(numeric(evidence_records[j].get("svlen", 0)))
            len_ratio = 1.0 if max(len_i, len_j) <= 0 else min(len_i, len_j) / max(len_i, len_j)
            if abs(pos_i - pos_j) <= 100 and len_ratio >= 0.7:
                node_i = i + 1
                node_j = j + 1
                edge_src.extend([node_i, node_j])
                edge_dst.extend([node_j, node_i])

    x = torch.tensor(node_features, dtype=torch.float32)
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    label = int(row.get("label", 0)) if not pd.isna(row.get("label", 0)) else 0
    graph = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float32))
    graph.metadata = {
        "candidate_id": str(row.get("candidate_id", "")),
        "chrom": str(row.get("chrom", "")),
        "start": int(numeric(row.get("start", 0))),
        "end": int(numeric(row.get("end", 0))),
        "svtype": str(row.get("svtype", "")),
        "median_svlen": int(numeric(row.get("median_svlen", 0))),
        "support_read_count": int(numeric(row.get("support_read_count", 0))),
    }
    graph.feature_names = FEATURE_NAMES
    return graph


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_parent_dir(args.out)

    if not os.path.exists(args.signals) or not os.path.exists(args.candidates):
        logging.warning("Missing input file. Writing an empty graph dataset.")
        torch.save([], args.out)
        return

    try:
        signals = pd.read_csv(args.signals, sep="\t")
        candidates = pd.read_csv(args.candidates, sep="\t")
    except pd.errors.EmptyDataError:
        logging.warning("Signals or candidates file has no readable rows. Writing an empty graph dataset.")
        torch.save([], args.out)
        return
    if signals.empty or candidates.empty:
        logging.warning("Signals or candidates are empty. Writing an empty graph dataset.")
        torch.save([], args.out)
        return

    signals = signals.copy()
    signals["event_pos"] = pd.to_numeric(signals["event_pos"], errors="coerce")
    signals["svlen"] = pd.to_numeric(signals["svlen"], errors="coerce")
    signals = signals.dropna(subset=["event_pos", "svlen"])
    signals["event_pos"] = signals["event_pos"].astype(int)

    if "label" not in candidates.columns:
        logging.warning("Candidate file has no label column; assigning label=0 to all graphs")
        candidates["label"] = 0

    graphs = [build_one_graph(signals, row) for _, row in candidates.iterrows()]
    torch.save(graphs, args.out)
    logging.info("Wrote %d graphs with %d features to %s", len(graphs), len(FEATURE_NAMES), args.out)


if __name__ == "__main__":
    main()
