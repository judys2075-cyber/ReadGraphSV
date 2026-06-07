#!/usr/bin/env python3
"""Run the ReadGraphSV v0.2 CIGAR plus extra-evidence inference pipeline."""

import argparse
import logging
import os
import subprocess
import sys

import pandas as pd


PREDICTION_FIELDS = [
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam", required=True, help="Input long-read BAM/CRAM/SAM")
    parser.add_argument("--model", required=True, help="Trained ReadGraphSV v0.2 GNN checkpoint")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="GNN probability threshold")
    parser.add_argument("--min_size", type=int, default=50, help="Minimum CIGAR I/D and soft-clip size")
    parser.add_argument("--cluster_window", type=int, default=500, help="Candidate clustering window")
    parser.add_argument("--min_support", type=int, default=1, help="Minimum candidate support reads")
    parser.add_argument("--extra_window", type=int, default=1000, help="Window for assigning extra evidence")
    parser.add_argument("--read_edge_window", type=int, default=100, help="Window for evidence-evidence graph edges")
    parser.add_argument("--truth", default=None, help="Optional truth VCF for labeling and evaluation")
    parser.add_argument("--max_dist", type=int, default=500, help="Maximum truth matching breakpoint distance")
    parser.add_argument("--min_size_sim", type=float, default=0.5, help="Minimum truth matching size similarity")
    return parser.parse_args(argv)


def script_path(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_step(name, command):
    logging.info("==== %s ====", name)
    logging.info("Command: %s", " ".join(command))
    subprocess.run(command, check=True)


def check_inputs(args):
    if not os.path.exists(args.bam):
        raise FileNotFoundError(f"Input BAM does not exist: {args.bam}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Input model does not exist: {args.model}")
    if args.truth and not os.path.exists(args.truth):
        raise FileNotFoundError(f"Truth VCF does not exist: {args.truth}")


def read_tsv(path, columns=None):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def write_unlabeled_candidates(candidates_path, out_path, label_value=0):
    candidates = read_tsv(candidates_path)
    if "label" not in candidates.columns:
        candidates["label"] = int(label_value)
    else:
        candidates["label"] = pd.to_numeric(candidates["label"], errors="coerce").fillna(label_value).astype(int)
    candidates.to_csv(out_path, sep="\t", index=False)
    logging.info("Prepared %d unlabeled candidates for graph construction: %s", len(candidates), out_path)


def read_predictions(path):
    return read_tsv(path, columns=PREDICTION_FIELDS)


def filter_predictions(predictions_path, out_path, threshold):
    pred = read_predictions(predictions_path)
    if pred.empty:
        pred.to_csv(out_path, sep="\t", index=False)
        logging.warning("No predictions found; wrote empty filtered table to %s", out_path)
        return 0

    if "gnn_prob" not in pred.columns:
        raise ValueError(f"prediction file has no gnn_prob column: {predictions_path}")

    pred = pred.copy()
    pred["gnn_prob"] = pd.to_numeric(pred["gnn_prob"], errors="coerce").fillna(0.0)
    if "gnn_pred" in pred.columns:
        pred["gnn_pred"] = pd.to_numeric(pred["gnn_pred"], errors="coerce").fillna(0).astype(int)
    else:
        pred["gnn_pred"] = 0
    filtered = pred[(pred["gnn_pred"] == 1) | (pred["gnn_prob"] >= threshold)]
    filtered.to_csv(out_path, sep="\t", index=False)
    logging.info("Filtered %d / %d candidates at threshold %.3f", len(filtered), len(pred), threshold)
    return len(filtered)


def chrom_sort_key(chrom):
    name = str(chrom)
    clean = name[3:] if name.lower().startswith("chr") else name
    special = {"x": 23, "y": 24, "m": 25, "mt": 25}
    lower = clean.lower()
    if clean.isdigit():
        return 0, int(clean), name
    if lower in special:
        return 0, special[lower], name
    return 1, clean, name


def numeric(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def estimate_svlen(row):
    median_svlen = abs(int(round(numeric(row.get("median_svlen", 0), 0))))
    if median_svlen > 0:
        return median_svlen
    start = int(round(numeric(row.get("start", 0), 0)))
    end = int(round(numeric(row.get("end", start + 1), start + 1)))
    return max(1, abs(end - start))


def write_vcf_header(handle):
    print("##fileformat=VCFv4.2", file=handle)
    print("##source=ReadGraphSV_v0.2", file=handle)
    print('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">', file=handle)
    print('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">', file=handle)
    print('##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="SV length">', file=handle)
    print('##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description="Supporting read count">', file=handle)
    print('##INFO=<ID=GNN_PROB,Number=1,Type=Float,Description="GNN confidence score">', file=handle)
    print('##ALT=<ID=DEL,Description="Deletion">', file=handle)
    print('##ALT=<ID=INS,Description="Insertion">', file=handle)
    print("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tReadGraphSV", file=handle)


def write_filtered_vcf(predictions_path, out_path, threshold):
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    pred = read_predictions(predictions_path)
    record_count = 0
    with open(out_path, "w") as handle:
        write_vcf_header(handle)
        if pred.empty:
            logging.warning("Prediction file is empty; wrote VCF header only: %s", out_path)
            return 0

        required = {"chrom", "start", "end", "svtype", "support_read_count", "gnn_prob"}
        missing = required - set(pred.columns)
        if missing:
            raise ValueError(f"prediction file is missing required columns: {sorted(missing)}")

        pred = pred.copy()
        pred["gnn_prob"] = pd.to_numeric(pred["gnn_prob"], errors="coerce").fillna(0.0)
        pred["start"] = pd.to_numeric(pred["start"], errors="coerce")
        pred["end"] = pd.to_numeric(pred["end"], errors="coerce")
        if "gnn_pred" in pred.columns:
            pred["gnn_pred"] = pd.to_numeric(pred["gnn_pred"], errors="coerce").fillna(0).astype(int)
        else:
            pred["gnn_pred"] = 0
        pred = pred.dropna(subset=["start", "end"])
        pred = pred[
            ((pred["gnn_pred"] == 1) | (pred["gnn_prob"] >= threshold))
            & (pred["svtype"].isin(["DEL", "INS"]))
        ].copy()
        if pred.empty:
            logging.info("No predictions passed threshold %.3f; wrote VCF header only", threshold)
            return 0

        pred["_chrom_key"] = pred["chrom"].map(chrom_sort_key)
        pred = pred.sort_values(["_chrom_key", "start", "end", "svtype"])

        for _, row in pred.iterrows():
            svtype = str(row["svtype"])
            start0 = int(round(numeric(row["start"], 0)))
            end0 = int(round(numeric(row["end"], start0 + 1)))
            svlen_abs = estimate_svlen(row)
            prob = numeric(row["gnn_prob"], 0.0)
            support = int(round(numeric(row["support_read_count"], 0)))

            if svtype == "DEL":
                alt = "<DEL>"
                vcf_end = max(end0, start0 + svlen_abs)
                svlen = -abs(svlen_abs)
            elif svtype == "INS":
                alt = "<INS>"
                vcf_end = start0 + 1
                svlen = abs(svlen_abs)
            else:
                continue

            record_count += 1
            info = (
                f"SVTYPE={svtype};END={vcf_end};SVLEN={svlen};"
                f"SUPPORT={support};GNN_PROB={prob:.6f}"
            )
            print(
                f"{row['chrom']}\t{start0 + 1}\tReadGraphSVv2_{record_count}\tN\t{alt}\t"
                f"{prob * 100:.2f}\tPASS\t{info}\tGT\t./.",
                file=handle,
            )

    logging.info("Wrote %d VCF records to %s", record_count, out_path)
    return record_count


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    check_inputs(args)

    outdir = os.path.abspath(args.outdir)
    data_dir = os.path.join(outdir, "data")
    graph_dir = os.path.join(outdir, "graphs")
    results_dir = os.path.join(outdir, "results")
    vcf_dir = os.path.join(outdir, "vcf")
    for path in [data_dir, graph_dir, results_dir, vcf_dir]:
        ensure_dir(path)

    signals = os.path.join(data_dir, "signals.tsv")
    candidates = os.path.join(data_dir, "candidates.tsv")
    extra_signals = os.path.join(data_dir, "extra_signals.tsv")
    candidates_labeled = os.path.join(data_dir, "candidates_labeled.tsv")
    candidates_for_graph = os.path.join(data_dir, "candidates_for_graph.tsv")
    graph_dataset = os.path.join(graph_dir, "dataset_v2.pt")
    predictions = os.path.join(results_dir, "predictions_v2.tsv")
    filtered_candidates = os.path.join(results_dir, "filtered_candidates.tsv")
    evaluation = os.path.join(results_dir, "evaluation_v2.txt")
    filtered_vcf = os.path.join(vcf_dir, "filtered.vcf")

    logging.info("ReadGraphSV v0.2 inference pipeline")
    logging.info("Output directory: %s", outdir)

    run_step(
        "Extract CIGAR DEL/INS signals",
        [
            sys.executable,
            script_path("extract_cigar_events.py"),
            "--bam",
            args.bam,
            "--min_size",
            str(args.min_size),
            "--out",
            signals,
        ],
    )

    run_step(
        "Cluster candidate SVs",
        [
            sys.executable,
            script_path("cluster_events.py"),
            "--signals",
            signals,
            "--window",
            str(args.cluster_window),
            "--min_support",
            str(args.min_support),
            "--out",
            candidates,
        ],
    )

    run_step(
        "Extract soft-clip, SA-tag, and supplementary evidence",
        [
            sys.executable,
            script_path("extract_extra_events.py"),
            "--bam",
            args.bam,
            "--min_clip",
            str(args.min_size),
            "--out",
            extra_signals,
        ],
    )

    if args.truth:
        run_step(
            "Label candidates with truth VCF",
            [
                sys.executable,
                script_path("label_candidates.py"),
                "--candidates",
                candidates,
                "--truth",
                args.truth,
                "--max_dist",
                str(args.max_dist),
                "--min_size_sim",
                str(args.min_size_sim),
                "--out",
                candidates_labeled,
            ],
        )
        labeled = read_tsv(candidates_labeled)
        labeled.to_csv(candidates_for_graph, sep="\t", index=False)
        logging.info("Using labeled candidates for graph construction: %s", candidates_for_graph)
    else:
        write_unlabeled_candidates(candidates, candidates_for_graph, label_value=0)
        logging.info("No truth VCF provided; continuing in unlabeled inference mode")

    run_step(
        "Build v0.2 graph dataset",
        [
            sys.executable,
            script_path("build_graph_dataset_v2.py"),
            "--signals",
            signals,
            "--extra",
            extra_signals,
            "--candidates",
            candidates_for_graph,
            "--out",
            graph_dataset,
            "--extra_window",
            str(args.extra_window),
            "--read_edge_window",
            str(args.read_edge_window),
        ],
    )

    run_step(
        "Predict with GNN",
        [
            sys.executable,
            script_path("predict_gnn.py"),
            "--dataset",
            graph_dataset,
            "--model",
            args.model,
            "--out",
            predictions,
            "--threshold",
            str(args.threshold),
        ],
    )

    filtered_count = filter_predictions(predictions, filtered_candidates, args.threshold)
    vcf_count = write_filtered_vcf(predictions, filtered_vcf, args.threshold)

    if args.truth:
        run_step(
            "Evaluate predictions",
            [
                sys.executable,
                script_path("evaluate_predictions.py"),
                "--pred",
                predictions,
                "--threshold",
                str(args.threshold),
                "--out",
                evaluation,
            ],
        )

    logging.info("ReadGraphSV v0.2 pipeline finished")
    logging.info("Signals: %s", signals)
    logging.info("Extra signals: %s", extra_signals)
    logging.info("Candidates: %s", candidates)
    logging.info("Candidates for graph: %s", candidates_for_graph)
    logging.info("Graph dataset: %s", graph_dataset)
    logging.info("Predictions: %s", predictions)
    logging.info("Filtered candidates: %s (%d records)", filtered_candidates, filtered_count)
    logging.info("Filtered VCF: %s (%d records)", filtered_vcf, vcf_count)
    if args.truth:
        logging.info("Labeled candidates: %s", candidates_labeled)
        logging.info("Evaluation: %s", evaluation)


if __name__ == "__main__":
    main()
