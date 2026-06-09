# Benchmark Notes

## Final HG002 chr21 Benchmark

The final ReadGraphSV v0.3 benchmark uses HG002 chr21 as held-out test data.
The threshold and deduplication parameters were selected on chr19 validation
data and then applied to chr21.

## Truvari Parameters

```bash
truvari bench \
  --includebed <Tier1 BED> \
  --passonly \
  --refdist 500 \
  --pctsize 0.5 \
  --sizemin 50 \
  --pctseq 0
```

The `--passonly` flag is important because non-PASS records in the truth VCF
should not be included in the denominator for this benchmark.

## Final Result Files

```text
results/final_hg002_chr21/readgraphsv_v3_final_comparison.tsv
results/final_hg002_chr21/README_final_results.md
```

## Interpretation

ReadGraphSV v0.3 + dedup achieved higher precision than Sniffles2 and higher
F1 than cuteSV and SVIM on the held-out chr21 DEL/INS benchmark. Sniffles2
remained the strongest overall caller in this comparison because it retained
the highest recall and F1.
