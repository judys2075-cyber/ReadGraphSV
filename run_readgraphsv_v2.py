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
    parser.add_argument(
        "--contig-length",
        type=int,
        default=None,
        help="Optional VCF contig length fallback when the BAM header cannot be read",
    )
    parser.add_argument("--max_dist", type=int, default=500, help="Maximum truth matching breakpoint distance")
    parser.add_argument("--min_size_sim", type=float, default=0.5, help="Minimum truth matching size similarity")
    parser.add_argument(
        "--use_extra_candidates",
        action="store_true",
        help="Enable v0.3 extra-evidence candidate proposal and CIGAR/extra candidate merging",
    )
    parser.add_argument("--extra_candidate_window", type=int, default=500, help="Window for v0.3 extra candidate clustering")
    parser.add_argument(
        "--min_softclip_support",
        type=int,
        default=10,
        help="Minimum SOFTCLIP support for v0.3 extra candidate proposal",
    )
    parser.add_argument(
        "--min_sa_support",
        type=int,
        default=2,
        help="Minimum SA_CONNECTION support for v0.3 extra candidate proposal",
    )
    parser.add_argument(
        "--min_supplementary_support",
        type=int,
        default=2,
        help="Minimum SUPPLEMENTARY support for v0.3 extra candidate proposal",
    )
    parser.add_argument(
        "--min_extra_only_support",
        type=int,
        default=30,
        help="Minimum support for EXTRA_ONLY candidates after v0.3 merging",
    )
    parser.add_argument(
        "--extra_candidate_min_size",
        type=int,
        default=None,
        help="Minimum size for v0.3 extra candidate proposal; defaults to --min_size",
    )
    parser.add_argument("--use_dedup", action="store_true", help="Enable VCF/candidate-level deduplication after GNN filtering")
    parser.add_argument("--dedup_window", type=int, default=500, help="Maximum start distance for deduplication")
    parser.add_argument("--dedup_min_size_sim", type=float, default=0.5, help="Minimum size similarity for deduplication")
    parser.add_argument("--dedup_score_col", default="gnn_prob", help="Score column used to choose dedup representatives")
    args = parser.parse_args(argv)
    if args.extra_candidate_min_size is None:
        args.extra_candidate_min_size = args.min_size
    return args


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


def count_tsv_rows(path):
    return len(read_tsv(path))


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


def prediction_chroms(pred):
    if pred.empty or "chrom" not in pred.columns:
        return []
    chroms = [str(chrom) for chrom in pred["chrom"].dropna().unique() if str(chrom)]
    return sorted(chroms, key=chrom_sort_key)


def read_bam_reference_lengths(bam_path):
    try:
        import pysam
    except ImportError:
        logging.warning("pysam is unavailable; VCF contig lengths will use fallback values")
        return {}

    try:
        with pysam.AlignmentFile(bam_path) as handle:
            return {str(name): int(length) for name, length in zip(handle.references, handle.lengths)}
    except (OSError, ValueError) as exc:
        logging.warning("Could not read BAM header from %s: %s", bam_path, exc)
        return {}


def lookup_contig_length(chrom, reference_lengths, fallback_length=None):
    chrom = str(chrom)
    for key in [chrom, chrom[3:] if chrom.lower().startswith("chr") else f"chr{chrom}"]:
        if key in reference_lengths:
            return int(reference_lengths[key])
    if fallback_length is not None:
        return int(fallback_length)
    return None


def write_contig_headers(handle, chroms, reference_lengths=None, fallback_length=None):
    reference_lengths = reference_lengths or {}
    for chrom in chroms:
        length = lookup_contig_length(chrom, reference_lengths, fallback_length=fallback_length)
        if length is None:
            print(f"##contig=<ID={chrom}>", file=handle)
        else:
            print(f"##contig=<ID={chrom},length={length}>", file=handle)


def write_vcf_header(handle, chroms=None, reference_lengths=None, fallback_contig_length=None):
    print("##fileformat=VCFv4.2", file=handle)
    print("##source=ReadGraphSV_v0.2", file=handle)
    write_contig_headers(handle, chroms or [], reference_lengths, fallback_contig_length)
    print('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">', file=handle)
    print('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">', file=handle)
    print('##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="SV length">', file=handle)
    print('##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description="Supporting read count">', file=handle)
    print('##INFO=<ID=GNN_PROB,Number=1,Type=Float,Description="GNN confidence score">', file=handle)
    print('##ALT=<ID=DEL,Description="Deletion">', file=handle)
    print('##ALT=<ID=INS,Description="Insertion">', file=handle)
    print('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">', file=handle)
    print("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tReadGraphSV", file=handle)


def write_filtered_vcf(predictions_path, out_path, threshold, bam_path=None, contig_length=None):
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    pred = read_predictions(predictions_path)
    reference_lengths = read_bam_reference_lengths(bam_path) if bam_path else {}
    chroms = prediction_chroms(pred)
    record_count = 0
    with open(out_path, "w") as handle:
        write_vcf_header(handle, chroms=chroms, reference_lengths=reference_lengths, fallback_contig_length=contig_length)
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
    extra_candidates = os.path.join(data_dir, "extra_candidates.tsv")
    candidates_v3_merged = os.path.join(data_dir, "candidates_v3_merged.tsv")
    candidates_labeled = os.path.join(data_dir, "candidates_labeled.tsv")
    candidates_for_graph = os.path.join(data_dir, "candidates_for_graph.tsv")
    graph_dataset = os.path.join(graph_dir, "dataset_v2.pt")
    predictions = os.path.join(results_dir, "predictions_v2.tsv")
    filtered_candidates = os.path.join(results_dir, "filtered_candidates.tsv")
    filtered_candidates_dedup = os.path.join(results_dir, "filtered_candidates_dedup.tsv")
    dedup_summary = os.path.join(results_dir, "dedup_summary.txt")
    evaluation = os.path.join(results_dir, "evaluation_v2.txt")
    filtered_vcf = os.path.join(vcf_dir, "filtered.vcf")
    filtered_dedup_vcf = os.path.join(vcf_dir, "filtered_dedup.vcf")

    logging.info("ReadGraphSV v0.2 inference pipeline")
    logging.info("Output directory: %s", outdir)
    logging.info("v0.3 extra candidate proposal enabled: %s", bool(args.use_extra_candidates))

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

    candidate_input = candidates
    if args.use_extra_candidates:
        logging.info("CIGAR candidates path: %s", candidates)
        logging.info("Extra candidates path: %s", extra_candidates)
        logging.info("Merged candidates path: %s", candidates_v3_merged)
        run_step(
            "Propose v0.3 extra-evidence candidates",
            [
                sys.executable,
                script_path("extra_candidate_proposer.py"),
                "--extra",
                extra_signals,
                "--out",
                extra_candidates,
                "--window",
                str(args.extra_candidate_window),
                "--min_size",
                str(args.extra_candidate_min_size),
                "--min_softclip_support",
                str(args.min_softclip_support),
                "--min_sa_support",
                str(args.min_sa_support),
                "--min_supplementary_support",
                str(args.min_supplementary_support),
            ],
        )
        run_step(
            "Merge CIGAR and v0.3 extra-evidence candidates",
            [
                sys.executable,
                script_path("merge_candidates_v3.py"),
                "--cigar-candidates",
                candidates,
                "--extra-candidates",
                extra_candidates,
                "--out",
                candidates_v3_merged,
                "--window",
                str(args.extra_candidate_window),
                "--min-size-sim",
                str(args.min_size_sim),
                "--min-extra-only-support",
                str(args.min_extra_only_support),
            ],
        )
        candidate_input = candidates_v3_merged
        logging.info(
            "v0.3 candidate counts: CIGAR=%d, extra=%d, merged=%d",
            count_tsv_rows(candidates),
            count_tsv_rows(extra_candidates),
            count_tsv_rows(candidates_v3_merged),
        )

    if args.truth:
        run_step(
            "Label candidates with truth VCF",
            [
                sys.executable,
                script_path("label_candidates.py"),
                "--candidates",
                candidate_input,
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
        write_unlabeled_candidates(candidate_input, candidates_for_graph, label_value=0)
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
    vcf_input = filtered_candidates
    final_vcf = filtered_vcf
    final_filtered_count = filtered_count

    if args.use_dedup:
        logging.info("Dedup enabled")
        run_step(
            "Deduplicate filtered candidates",
            [
                sys.executable,
                script_path("dedup_filtered_candidates.py"),
                "--filtered",
                filtered_candidates,
                "--labeled",
                candidates_labeled,
                "--out",
                filtered_candidates_dedup,
                "--window",
                str(args.dedup_window),
                "--min-size-sim",
                str(args.dedup_min_size_sim),
                "--score-col",
                args.dedup_score_col,
            ],
        )
        dedup_count = count_tsv_rows(filtered_candidates_dedup)
        removed_count = max(0, filtered_count - dedup_count)
        logging.info("Dedup input candidates: %d", filtered_count)
        logging.info("Dedup output candidates: %d", dedup_count)
        logging.info("Removed candidates: %d", removed_count)
        vcf_input = filtered_candidates_dedup
        final_vcf = filtered_dedup_vcf
        final_filtered_count = dedup_count

    vcf_count = write_filtered_vcf(
        vcf_input,
        final_vcf,
        args.threshold,
        bam_path=args.bam,
        contig_length=args.contig_length,
    )

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

    candidate_count = count_tsv_rows(candidate_input)

    logging.info("ReadGraphSV v0.2 pipeline finished")
    logging.info("==== ReadGraphSV run summary ====")
    logging.info("Candidate input: %s (%d candidates)", candidate_input, candidate_count)
    logging.info("Filtered candidates: %s (%d candidates)", filtered_candidates, filtered_count)
    if args.use_dedup:
        logging.info("Deduplicated candidates: %s (%d candidates)", filtered_candidates_dedup, final_filtered_count)
    else:
        logging.info("Deduplicated candidates: not enabled")
    logging.info("Final VCF: %s (%d records)", final_vcf, vcf_count)
    logging.info("Signals: %s", signals)
    logging.info("Extra signals: %s", extra_signals)
    logging.info("Candidates: %s", candidates)
    if args.use_extra_candidates:
        logging.info("Extra candidates: %s", extra_candidates)
        logging.info("v0.3 merged candidates: %s", candidates_v3_merged)
    logging.info("Candidates for graph: %s", candidates_for_graph)
    logging.info("Graph dataset: %s", graph_dataset)
    logging.info("Predictions: %s", predictions)
    logging.info("Filtered candidates: %s (%d records)", filtered_candidates, filtered_count)
    if args.use_dedup:
        logging.info("Deduplicated filtered candidates: %s (%d records)", filtered_candidates_dedup, final_filtered_count)
        logging.info("Dedup summary: %s", dedup_summary)
    logging.info("Filtered VCF: %s (%d records)", final_vcf, vcf_count)
    logging.info("Final VCF path: %s", final_vcf)
    if args.truth:
        logging.info("Labeled candidates: %s", candidates_labeled)
        logging.info("Evaluation: %s", evaluation)


if __name__ == "__main__":
    main()
