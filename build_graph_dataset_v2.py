#!/usr/bin/env python3
"""Build v0.2 ReadGraphSV graphs with CIGAR and extra evidence nodes."""

import argparse
import logging
import os

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from utils_graph import bool_numeric, log1p_norm, numeric, strand_numeric


EVIDENCE_TYPES = [
    "candidate",
    "cigar_del",
    "cigar_ins",
    "softclip_left",
    "softclip_right",
    "sa_connection",
    "supplementary",
]

FEATURE_NAMES_V2 = [
    "svtype_DEL",
    "svtype_INS",
    "log_length",
    "log_support_or_read_len",
    "mapq_norm",
    "strand_numeric",
    "distance_to_candidate_center_norm",
    "has_sa",
    "is_supplementary",
    "chrom_change",
    "orientation_change",
    *[f"is_{name}" for name in EVIDENCE_TYPES],
]

MAX_EVENT_LEN = 1_000_000.0
DISTANCE_SCALE = 1000.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", required=True, help="Input CIGAR signals.tsv")
    parser.add_argument("--extra", required=True, help="Input extra_signals.tsv")
    parser.add_argument("--candidates", required=True, help="Input candidates_labeled.tsv")
    parser.add_argument("--out", required=True, help="Output graphs/dataset_v2.pt")
    parser.add_argument("--extra_window", type=int, default=1000, help="Window for assigning extra evidence")
    parser.add_argument("--read_edge_window", type=int, default=100, help="Window for connecting evidence nodes")
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def empty_signals_df():
    return pd.DataFrame(columns=["chrom", "event_pos", "svtype", "svlen"])


def empty_extra_df():
    return pd.DataFrame(columns=["evidence_type", "src_chrom", "event_pos", "dst_chrom", "dst_start"])


def safe_read_table(path, empty_factory):
    if not os.path.exists(path):
        logging.warning("Input file not found: %s; using an empty table", path)
        return empty_factory()
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        logging.warning("Input file has no readable rows: %s; using an empty table", path)
        return empty_factory()


def prepare_numeric_columns(df, columns):
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return df


def svtype_flags(svtype):
    text = str(svtype)
    return 1.0 if text == "DEL" else 0.0, 1.0 if text == "INS" else 0.0


def evidence_one_hot(evidence_type):
    return [1.0 if evidence_type == item else 0.0 for item in EVIDENCE_TYPES]


def clipped_log_length(value):
    return log1p_norm(min(abs(numeric(value)), MAX_EVENT_LEN), scale=MAX_EVENT_LEN)


def mapq_norm(value):
    return min(1.0, max(0.0, numeric(value)) / 60.0)


def distance_norm(position, candidate_center):
    return min(1.0, abs(numeric(position) - candidate_center) / DISTANCE_SCALE)


def candidate_center(row):
    start = numeric(row.get("start", 0))
    end = numeric(row.get("end", start))
    return (start + end) / 2.0


def candidate_anchors(row):
    start = numeric(row.get("start", 0))
    end = numeric(row.get("end", start))
    return [start, end, (start + end) / 2.0]


def label_from_row(row):
    if "label" not in row or pd.isna(row.get("label", 0)):
        return 0
    return int(numeric(row.get("label", 0)))


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


def signals_for_candidate(signals, row):
    indices = parse_signal_indices(row.get("signal_indices", ""))
    if indices:
        return signals.loc[signals.index.intersection(indices)].copy()

    chrom = str(row.get("chrom", ""))
    svtype = str(row.get("svtype", ""))
    start = int(numeric(row.get("start", 0)))
    end = int(numeric(row.get("end", start)))
    margin = 500
    if signals.empty:
        return signals.copy()
    return signals[
        (signals["chrom"].astype(str) == chrom)
        & (signals["svtype"].astype(str) == svtype)
        & (signals["event_pos"].between(start - margin, end + margin))
    ].copy()


def distance_to_any_anchor(position, anchors):
    return min(abs(numeric(position) - anchor) for anchor in anchors)


def best_candidate_position(row, candidate_chrom, anchors, window):
    """Return the evidence position to use in this graph, or None if unmatched."""
    positions = []
    if str(row.get("src_chrom", "")) == candidate_chrom:
        src_pos = numeric(row.get("event_pos", 0))
        if distance_to_any_anchor(src_pos, anchors) <= window:
            positions.append(src_pos)

    if str(row.get("evidence_type", "")) == "SA_CONNECTION" and str(row.get("dst_chrom", "")) == candidate_chrom:
        dst_start = numeric(row.get("dst_start", 0))
        if distance_to_any_anchor(dst_start, anchors) <= window:
            positions.append(dst_start)

    if not positions:
        return None

    center = anchors[-1]
    return min(positions, key=lambda pos: abs(pos - center))


def extra_for_candidate(extra, row, extra_window):
    if extra.empty:
        return []

    chrom = str(row.get("chrom", ""))
    anchors = candidate_anchors(row)
    records = []
    for _, extra_row in extra.iterrows():
        graph_pos = best_candidate_position(extra_row, chrom, anchors, extra_window)
        if graph_pos is None:
            continue
        record = extra_row.to_dict()
        record["_graph_event_pos"] = graph_pos
        records.append(record)
    return records


def make_feature(
    svtype,
    length,
    support_or_read_len,
    mapq,
    strand,
    graph_event_pos,
    center,
    has_sa,
    is_supplementary,
    chrom_change,
    orientation_change,
    evidence_type,
):
    sv_del, sv_ins = svtype_flags(svtype)
    return [
        sv_del,
        sv_ins,
        clipped_log_length(length),
        log1p_norm(max(0.0, numeric(support_or_read_len)), scale=100000.0),
        mapq_norm(mapq),
        strand_numeric(strand),
        distance_norm(graph_event_pos, center),
        bool_numeric(has_sa),
        bool_numeric(is_supplementary),
        bool_numeric(chrom_change),
        bool_numeric(orientation_change),
        *evidence_one_hot(evidence_type),
    ]


def candidate_feature(row, center):
    return make_feature(
        svtype=row.get("svtype", ""),
        length=row.get("median_svlen", 0),
        support_or_read_len=row.get("support_read_count", 0),
        mapq=row.get("mean_mapq", 0),
        strand="",
        graph_event_pos=center,
        center=center,
        has_sa=0,
        is_supplementary=0,
        chrom_change=0,
        orientation_change=0,
        evidence_type="candidate",
    )


def cigar_feature(signal, center):
    svtype = str(signal.get("svtype", ""))
    evidence_type = "cigar_del" if svtype == "DEL" else "cigar_ins"
    return make_feature(
        svtype=svtype,
        length=signal.get("svlen", 0),
        support_or_read_len=signal.get("read_len", 0),
        mapq=signal.get("mapq", 0),
        strand=signal.get("strand", ""),
        graph_event_pos=signal.get("event_pos", 0),
        center=center,
        has_sa=signal.get("has_sa", 0),
        is_supplementary=signal.get("is_supplementary", 0),
        chrom_change=0,
        orientation_change=0,
        evidence_type=evidence_type,
    )


def extra_feature(extra_row, candidate_svtype, center):
    evidence_map = {
        "SOFTCLIP_LEFT": "softclip_left",
        "SOFTCLIP_RIGHT": "softclip_right",
        "SA_CONNECTION": "sa_connection",
        "SUPPLEMENTARY": "supplementary",
    }
    evidence_type = evidence_map.get(str(extra_row.get("evidence_type", "")), "supplementary")
    return make_feature(
        svtype=candidate_svtype,
        length=extra_row.get("event_len", 0),
        support_or_read_len=0,
        mapq=extra_row.get("mapq", 0),
        strand=extra_row.get("src_strand", ""),
        graph_event_pos=extra_row.get("_graph_event_pos", extra_row.get("event_pos", 0)),
        center=center,
        has_sa=extra_row.get("has_sa", 0),
        is_supplementary=extra_row.get("is_supplementary", 0),
        chrom_change=extra_row.get("chrom_change", 0),
        orientation_change=extra_row.get("orientation_change", 0),
        evidence_type=evidence_type,
    )


def add_bidirectional_edge(edge_src, edge_dst, left, right):
    edge_src.extend([left, right])
    edge_dst.extend([right, left])


def build_edges(evidence_positions, read_edge_window):
    edge_src = []
    edge_dst = []

    for node_idx in range(1, len(evidence_positions) + 1):
        add_bidirectional_edge(edge_src, edge_dst, 0, node_idx)

    for i in range(len(evidence_positions)):
        for j in range(i + 1, len(evidence_positions)):
            if abs(numeric(evidence_positions[i]) - numeric(evidence_positions[j])) <= read_edge_window:
                add_bidirectional_edge(edge_src, edge_dst, i + 1, j + 1)

    if edge_src:
        return torch.tensor([edge_src, edge_dst], dtype=torch.long)
    return torch.empty((2, 0), dtype=torch.long)


def build_one_graph(signals, extra, row, extra_window, read_edge_window):
    center = candidate_center(row)
    candidate_svtype = str(row.get("svtype", ""))
    cigar_evidence = signals_for_candidate(signals, row)
    extra_evidence = extra_for_candidate(extra, row, extra_window)

    node_features = [candidate_feature(row, center)]
    evidence_positions = []

    for _, signal in cigar_evidence.iterrows():
        node_features.append(cigar_feature(signal, center))
        evidence_positions.append(numeric(signal.get("event_pos", center)))

    for extra_row in extra_evidence:
        node_features.append(extra_feature(extra_row, candidate_svtype, center))
        evidence_positions.append(numeric(extra_row.get("_graph_event_pos", center)))

    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = build_edges(evidence_positions, read_edge_window)
    graph = Data(x=x, edge_index=edge_index, y=torch.tensor([label_from_row(row)], dtype=torch.float32))

    graph.metadata = {
        "candidate_id": str(row.get("candidate_id", "")),
        "chrom": str(row.get("chrom", "")),
        "start": int(numeric(row.get("start", 0))),
        "end": int(numeric(row.get("end", 0))),
        "svtype": candidate_svtype,
        "support_read_count": int(numeric(row.get("support_read_count", 0))),
        "median_svlen": int(numeric(row.get("median_svlen", 0))),
        "num_cigar_nodes": int(len(cigar_evidence)),
        "num_extra_nodes": int(len(extra_evidence)),
    }
    graph.feature_names = FEATURE_NAMES_V2
    return graph


def prepare_signals(signals):
    signals = prepare_numeric_columns(signals, ["event_pos", "svlen", "read_len", "mapq"])
    if "chrom" not in signals.columns:
        signals["chrom"] = ""
    if "svtype" not in signals.columns:
        signals["svtype"] = ""
    return signals.dropna(subset=["event_pos"]).copy()


def prepare_extra(extra):
    extra = prepare_numeric_columns(
        extra,
        [
            "event_pos",
            "event_len",
            "mapq",
            "dst_start",
            "has_sa",
            "is_supplementary",
            "chrom_change",
            "orientation_change",
        ],
    )
    for column in ["evidence_type", "src_chrom", "dst_chrom", "src_strand"]:
        if column not in extra.columns:
            extra[column] = ""
    return extra.copy()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_parent_dir(args.out)

    candidates = safe_read_table(args.candidates, lambda: pd.DataFrame())
    if candidates.empty:
        logging.warning("No candidates found. Writing an empty graph dataset.")
        torch.save([], args.out)
        return

    if "label" not in candidates.columns:
        logging.warning("Candidate file has no label column; assigning label=0 to all graphs")
        candidates["label"] = 0

    signals = prepare_signals(safe_read_table(args.signals, empty_signals_df))
    extra = prepare_extra(safe_read_table(args.extra, empty_extra_df))

    graphs = []
    cigar_counts = []
    extra_counts = []
    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="graphs", unit="graph"):
        graph = build_one_graph(signals, extra, row, args.extra_window, args.read_edge_window)
        graphs.append(graph)
        cigar_counts.append(graph.metadata["num_cigar_nodes"])
        extra_counts.append(graph.metadata["num_extra_nodes"])

    torch.save(graphs, args.out)
    avg_cigar = sum(cigar_counts) / len(cigar_counts) if cigar_counts else 0.0
    avg_extra = sum(extra_counts) / len(extra_counts) if extra_counts else 0.0
    no_extra = sum(1 for count in extra_counts if count == 0)
    logging.info("Wrote %d v0.2 graphs with %d features to %s", len(graphs), len(FEATURE_NAMES_V2), args.out)
    logging.info("Average CIGAR nodes per graph: %.3f", avg_cigar)
    logging.info("Average extra nodes per graph: %.3f", avg_extra)
    logging.info("Candidates with no extra nodes: %d", no_extra)


if __name__ == "__main__":
    main()
