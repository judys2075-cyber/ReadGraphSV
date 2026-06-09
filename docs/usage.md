# ReadGraphSV Usage Notes

## Recommended v0.3 Command

```bash
python run_readgraphsv_v2.py \
  --bam real_data/HG002_chr21/bam/HG002_chr21.bam \
  --model models/readgraph_gnn_v3_extra_candidates_chr20_chr22.pt \
  --outdir runs/HG002_chr21_v3 \
  --truth real_data/HG002_chr21/truth_chr21/HG002_chr21_DELINS_50.vcf.gz \
  --threshold 0.65 \
  --min_size 50 \
  --cluster_window 500 \
  --min_support 1 \
  --extra_window 1000 \
  --read_edge_window 100 \
  --max_dist 500 \
  --min_size_sim 0.5 \
  --use_extra_candidates \
  --extra_candidate_window 500 \
  --min_softclip_support 10 \
  --min_sa_support 2 \
  --min_supplementary_support 2 \
  --min_extra_only_support 30 \
  --use_dedup \
  --dedup_window 500 \
  --dedup_min_size_sim 0.5
```

## Inference Without Truth

For unlabeled inference, omit `--truth`. The wrapper will add a placeholder
label column for graph construction and skip the final evaluation report.

## Output Selection

Without `--use_dedup`, the final VCF is:

```text
outdir/vcf/filtered.vcf
```

With `--use_dedup`, the final VCF is:

```text
outdir/vcf/filtered_dedup.vcf
```

The deduplication report is:

```text
outdir/results/dedup_summary.txt
```
