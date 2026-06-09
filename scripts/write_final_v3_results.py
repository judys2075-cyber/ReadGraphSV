#!/usr/bin/env python3
"""Write the final ReadGraphSV v0.3 HG002 chr21 benchmark table and README."""

import argparse
import logging
from pathlib import Path

try:
    from benchmark_truvari_delins import read_truvari_summary
except ImportError:  # pragma: no cover - used when imported as scripts.write_final_v3_results
    from scripts.benchmark_truvari_delins import read_truvari_summary


RESULT_FIELDS = ["Tool", "Precision", "Recall", "F1", "TP-comp", "TP-base", "FP", "FN", "Summary"]
DEFAULT_OUTDIR = "results/final_hg002_chr21"
DEFAULT_TRUTH = "real_data/HG002_chr21/truth_chr21/HG002_chr21_DELINS_50.vcf.gz"
DEFAULT_BED = "real_data/HG002_chr21/truth_chr21/HG002_chr21_Tier1.bed"
DEFAULT_SUMMARIES = [
    (
        "ReadGraphSV v0.2 real-finetuned",
        "results/reproduce_hg002_chr21_benchmark/truvari_readgraphsv/summary.json",
    ),
    (
        "ReadGraphSV v0.3 trained",
        "results/v3_trained_model_chr21_benchmark_t065/truvari_readgraphsv/summary.json",
    ),
    (
        "ReadGraphSV v0.3 trained + dedup",
        "results/v3_dedup_chr21_benchmark_w500_s05/truvari_readgraphsv/summary.json",
    ),
    ("Sniffles2", "results/reproduce_hg002_chr21_benchmark/truvari_sniffles2/summary.json"),
    ("cuteSV", "results/reproduce_hg002_chr21_benchmark/truvari_cutesv/summary.json"),
    ("SVIM", "results/reproduce_hg002_chr21_benchmark/truvari_svim/summary.json"),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readgraphsv-v02-summary",
        default=DEFAULT_SUMMARIES[0][1],
        help="ReadGraphSV v0.2 real-finetuned Truvari summary.json",
    )
    parser.add_argument(
        "--readgraphsv-v03-summary",
        default=DEFAULT_SUMMARIES[1][1],
        help="ReadGraphSV v0.3 trained Truvari summary.json",
    )
    parser.add_argument(
        "--readgraphsv-v03-dedup-summary",
        default=DEFAULT_SUMMARIES[2][1],
        help="ReadGraphSV v0.3 trained + dedup Truvari summary.json",
    )
    parser.add_argument("--sniffles2-summary", default=DEFAULT_SUMMARIES[3][1], help="Sniffles2 Truvari summary.json")
    parser.add_argument("--cutesv-summary", default=DEFAULT_SUMMARIES[4][1], help="cuteSV Truvari summary.json")
    parser.add_argument("--svim-summary", default=DEFAULT_SUMMARIES[5][1], help="SVIM Truvari summary.json")
    parser.add_argument("--truth-vcf", default=DEFAULT_TRUTH, help="Truth VCF path to document in the final README")
    parser.add_argument("--tier1-bed", default=DEFAULT_BED, help="Tier1 confident BED path to document in the final README")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Final result output directory")
    return parser.parse_args(argv)


def summary_inputs(args):
    return [
        ("ReadGraphSV v0.2 real-finetuned", args.readgraphsv_v02_summary),
        ("ReadGraphSV v0.3 trained", args.readgraphsv_v03_summary),
        ("ReadGraphSV v0.3 trained + dedup", args.readgraphsv_v03_dedup_summary),
        ("Sniffles2", args.sniffles2_summary),
        ("cuteSV", args.cutesv_summary),
        ("SVIM", args.svim_summary),
    ]


def format_float(value):
    return f"{float(value):.6f}"


def load_result_rows(args):
    rows = []
    for label, summary_path in summary_inputs(args):
        if not Path(summary_path).exists():
            raise FileNotFoundError(f"Required Truvari summary for {label} does not exist: {summary_path}")
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
    return rows


def write_result_tsv(rows, output_path):
    with open(output_path, "w") as handle:
        handle.write("\t".join(RESULT_FIELDS) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(field, "")) for field in RESULT_FIELDS) + "\n")


def write_markdown_table(rows, handle):
    print("| Tool | Precision | Recall | F1 | TP-comp | TP-base | FP | FN |", file=handle)
    print("|---|---:|---:|---:|---:|---:|---:|---:|", file=handle)
    for row in rows:
        print(
            f"| {row['Tool']} | {row['Precision']} | {row['Recall']} | {row['F1']} | "
            f"{row['TP-comp']} | {row['TP-base']} | {row['FP']} | {row['FN']} |",
            file=handle,
        )


def write_result_readme(rows, output_path, truth_vcf, tier1_bed):
    with open(output_path, "w") as handle:
        print("# ReadGraphSV v0.3 Final HG002 chr21 Results", file=handle)
        print("", file=handle)
        print("## Dataset", file=handle)
        print("", file=handle)
        print("- Dataset: HG002 chr21 held-out benchmark", file=handle)
        print("- Read data: HG002 PacBio HiFi Revio aligned to GRCh37", file=handle)
        print("- SV types: DEL/INS", file=handle)
        print("- Minimum SV size: 50 bp", file=handle)
        print(f"- Truth VCF: `{truth_vcf}`", file=handle)
        print(f"- Tier1 BED: `{tier1_bed}`", file=handle)
        print("", file=handle)
        print("## Truvari Benchmark Parameters", file=handle)
        print("", file=handle)
        print("```bash", file=handle)
        print("truvari bench \\", file=handle)
        print("  --includebed <Tier1 BED> \\", file=handle)
        print("  --passonly \\", file=handle)
        print("  --refdist 500 \\", file=handle)
        print("  --pctsize 0.5 \\", file=handle)
        print("  --sizemin 50 \\", file=handle)
        print("  --pctseq 0", file=handle)
        print("```", file=handle)
        print("", file=handle)
        print("## ReadGraphSV v0.3 Recommended Command", file=handle)
        print("", file=handle)
        print("```bash", file=handle)
        print("python run_readgraphsv_v2.py \\", file=handle)
        print("  --bam real_data/HG002_chr21/bam/HG002_chr21.bam \\", file=handle)
        print("  --model models/readgraph_gnn_v3.pt \\", file=handle)
        print("  --truth real_data/HG002_chr21/truth_chr21/HG002_chr21_DELINS_50.vcf.gz \\", file=handle)
        print("  --outdir runs/HG002_chr21_v3_final \\", file=handle)
        print("  --threshold 0.65 \\", file=handle)
        print("  --use_extra_candidates \\", file=handle)
        print("  --use_dedup \\", file=handle)
        print("  --dedup_window 500 \\", file=handle)
        print("  --dedup_min_size_sim 0.5", file=handle)
        print("```", file=handle)
        print("", file=handle)
        print("## Validation and Test Protocol", file=handle)
        print("", file=handle)
        print("- Threshold source: chr19 validation selected `threshold=0.65`.", file=handle)
        print("- Deduplication source: chr19 validation showed `window=500` and `min-size-sim=0.5` were effective.", file=handle)
        print("- Held-out policy: chr21 final benchmark was used only as the held-out test and was not used for tuning.", file=handle)
        print("", file=handle)
        print("## Final Comparison", file=handle)
        print("", file=handle)
        write_markdown_table(rows, handle)
        print("", file=handle)
        print("## Conclusion", file=handle)
        print("", file=handle)
        print(
            "ReadGraphSV v0.3 + dedup achieved F1=0.9107 on HG002 chr21, outperforming cuteSV and "
            "SVIM in F1 and achieving higher precision than Sniffles2, while Sniffles2 retained the "
            "highest recall and overall F1.",
            file=handle,
        )


def write_final_v3_results(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_result_rows(args)
    table_path = outdir / "readgraphsv_v3_final_comparison.tsv"
    readme_path = outdir / "README_final_results.md"
    write_result_tsv(rows, table_path)
    write_result_readme(rows, readme_path, args.truth_vcf, args.tier1_bed)
    return table_path, readme_path


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    table_path, readme_path = write_final_v3_results(args)
    logging.info("Wrote ReadGraphSV v0.3 final comparison table to %s", table_path)
    logging.info("Wrote ReadGraphSV v0.3 final README to %s", readme_path)


if __name__ == "__main__":
    main()
