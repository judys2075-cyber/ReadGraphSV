#!/usr/bin/env python3
"""Write final HG002 chr21 benchmark tables from Truvari summary.json files."""

import argparse
import logging
from pathlib import Path

try:
    from benchmark_truvari_delins import read_truvari_summary
except ImportError:  # pragma: no cover - used when imported as scripts.write_final_hg002_results
    from scripts.benchmark_truvari_delins import read_truvari_summary


RESULT_FIELDS = ["Tool", "Precision", "Recall", "F1", "TP-comp", "TP-base", "FP", "FN", "Summary"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readgraphsv-sim-summary", help="Simulation-trained ReadGraphSV Truvari summary.json")
    parser.add_argument("--readgraphsv-finetuned-summary", help="Real-finetuned ReadGraphSV Truvari summary.json")
    parser.add_argument("--sniffles2-summary", help="Sniffles2 Truvari summary.json")
    parser.add_argument("--cutesv-summary", help="cuteSV Truvari summary.json")
    parser.add_argument("--svim-summary", help="SVIM Truvari summary.json")
    parser.add_argument("--outdir", default="results/final_hg002_chr21", help="Final result output directory")
    return parser.parse_args(argv)


def summary_inputs(args):
    return [
        ("ReadGraphSV simulation-trained", args.readgraphsv_sim_summary),
        ("ReadGraphSV real-finetuned", args.readgraphsv_finetuned_summary),
        ("Sniffles2", args.sniffles2_summary),
        ("cuteSV", args.cutesv_summary),
        ("SVIM", args.svim_summary),
    ]


def format_float(value):
    return f"{float(value):.6f}"


def load_result_rows(args):
    rows = []
    for label, summary_path in summary_inputs(args):
        if not summary_path:
            logging.warning("No summary path provided for %s; skipping", label)
            continue
        if not Path(summary_path).exists():
            raise FileNotFoundError(f"Summary for {label} does not exist: {summary_path}")
        metrics = read_truvari_summary(summary_path)
        rows.append(
            {
                "Tool": label,
                "Precision": format_float(metrics["Precision"]),
                "Recall": format_float(metrics["Recall"]),
                "F1": format_float(metrics["F1"]),
                "TP-comp": metrics["TP-comp"],
                "TP-base": metrics["TP-base"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "Summary": summary_path,
            }
        )
    if not rows:
        raise ValueError("No Truvari summary files were provided")
    return rows


def write_result_tsv(rows, output_path):
    with open(output_path, "w") as handle:
        handle.write("\t".join(RESULT_FIELDS) + "\n")
        for row in rows:
            values = [str(row.get(field, "")) for field in RESULT_FIELDS]
            if len(values) != len(RESULT_FIELDS):
                raise ValueError("Final result row does not match header width")
            handle.write("\t".join(values) + "\n")


def write_result_readme(rows, output_path):
    with open(output_path, "w") as handle:
        print("# Final HG002 chr21 DEL/INS Benchmark", file=handle)
        print("", file=handle)
        print("- Dataset: HG002 PacBio HiFi Revio GRCh37", file=handle)
        print("- SV types: DEL/INS", file=handle)
        print("- Minimum size: SVLEN >= 50", file=handle)
        print("- Truth set: GIAB/NIST HG002 Tier1 v0.6", file=handle)
        print("- Evaluation: Truvari evaluation", file=handle)
        print("- Real-finetuned model: trained on chr20+chr22", file=handle)
        print("- Threshold selection: chr19", file=handle)
        print("- Final test: held-out chr21", file=handle)
        print("", file=handle)
        print("| Tool | Precision | Recall | F1 | TP-comp | TP-base | FP | FN |", file=handle)
        print("|---|---:|---:|---:|---:|---:|---:|---:|", file=handle)
        for row in rows:
            print(
                f"| {row['Tool']} | {row['Precision']} | {row['Recall']} | {row['F1']} | "
                f"{row['TP-comp']} | {row['TP-base']} | {row['FP']} | {row['FN']} |",
                file=handle,
            )


def write_final_results(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_result_rows(args)
    table_path = outdir / "readgraphsv_vs_tools_final.tsv"
    readme_path = outdir / "README_final_results.md"
    write_result_tsv(rows, table_path)
    write_result_readme(rows, readme_path)
    return table_path, readme_path


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    table_path, readme_path = write_final_results(args)
    logging.info("Wrote final result table to %s", table_path)
    logging.info("Wrote final result README to %s", readme_path)


if __name__ == "__main__":
    main()
