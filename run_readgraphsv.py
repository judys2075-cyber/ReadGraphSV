#!/usr/bin/env python3
"""Run the ReadGraphSV v0.1 CIGAR DEL/INS inference pipeline end to end."""

import argparse
import logging
import os
import shutil
import subprocess
import sys

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam", required=True, help="Input long-read BAM/CRAM/SAM")
    parser.add_argument("--model", required=True, help="Trained ReadGraphSV GNN checkpoint")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="GNN probability threshold")
    parser.add_argument("--min_size", type=int, default=50, help="Minimum CIGAR I/D size")
    parser.add_argument("--window", type=int, default=500, help="Candidate clustering window")
    parser.add_argument("--min_support", type=int, default=1, help="Minimum candidate support reads")
    parser.add_argument("--truth", default=None, help="Optional truth VCF for evaluation mode")
    parser.add_argument("--no_vcf", action="store_true", help="Do not export filtered VCF")
    return parser.parse_args()


def script_path(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


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


def read_predictions(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_filtered_predictions(predictions_path, out_path, threshold):
    pred = read_predictions(predictions_path)
    if pred.empty:
        pred.to_csv(out_path, sep="\t", index=False)
        logging.warning("No predictions found; wrote empty filtered table to %s", out_path)
        return 0
    if "gnn_prob" not in pred.columns:
        raise ValueError(f"prediction file has no gnn_prob column: {predictions_path}")
    pred = pred.copy()
    pred["gnn_prob"] = pd.to_numeric(pred["gnn_prob"], errors="coerce").fillna(0.0)
    filtered = pred[pred["gnn_prob"] >= threshold]
    filtered.to_csv(out_path, sep="\t", index=False)
    logging.info("Filtered %d / %d candidates at threshold %.3f", len(filtered), len(pred), threshold)
    return len(filtered)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    check_inputs(args)
    os.makedirs(args.outdir, exist_ok=True)

    outdir = os.path.abspath(args.outdir)
    signals = os.path.join(outdir, "signals.tsv")
    candidates = os.path.join(outdir, "candidates.tsv")
    candidates_labeled = os.path.join(outdir, "candidates_labeled.tsv")
    candidates_for_graph = os.path.join(outdir, "candidates_for_graph.tsv")
    graph_dataset = os.path.join(outdir, "graph_dataset.pt")
    predictions = os.path.join(outdir, "predictions.tsv")
    filtered_candidates = os.path.join(outdir, "filtered_candidates.tsv")
    filtered_vcf = os.path.join(outdir, "filtered.vcf")
    evaluation = os.path.join(outdir, "evaluation.txt")

    logging.info("ReadGraphSV v0.1 inference pipeline")
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
            str(args.window),
            "--min_support",
            str(args.min_support),
            "--out",
            candidates,
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
                "--out",
                candidates_labeled,
            ],
        )
        shutil.copyfile(candidates_labeled, candidates_for_graph)
    else:
        shutil.copyfile(candidates, candidates_for_graph)
        logging.info("No truth VCF provided; using unlabeled candidates for graph construction")

    run_step(
        "Build graph dataset",
        [
            sys.executable,
            script_path("build_graph_dataset.py"),
            "--signals",
            signals,
            "--candidates",
            candidates_for_graph,
            "--out",
            graph_dataset,
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

    filtered_count = write_filtered_predictions(predictions, filtered_candidates, args.threshold)

    if not args.no_vcf:
        run_step(
            "Export filtered VCF",
            [
                sys.executable,
                script_path("export_vcf.py"),
                "--pred",
                predictions,
                "--out",
                filtered_vcf,
                "--threshold",
                str(args.threshold),
            ],
        )
    else:
        logging.info("--no_vcf set; skipping VCF export")

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

    logging.info("ReadGraphSV pipeline finished")
    logging.info("Signals: %s", signals)
    logging.info("Candidates: %s", candidates)
    logging.info("Graph dataset: %s", graph_dataset)
    logging.info("Predictions: %s", predictions)
    logging.info("Filtered candidates: %s (%d records)", filtered_candidates, filtered_count)
    if not args.no_vcf:
        logging.info("Filtered VCF: %s", filtered_vcf)
    if args.truth:
        logging.info("Labeled candidates: %s", candidates_labeled)
        logging.info("Evaluation: %s", evaluation)


if __name__ == "__main__":
    main()
