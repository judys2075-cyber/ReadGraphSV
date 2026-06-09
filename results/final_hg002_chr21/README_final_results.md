# ReadGraphSV v0.3 Final HG002 chr21 Results

## Dataset

- Dataset: HG002 chr21 held-out benchmark
- Read data: HG002 PacBio HiFi Revio aligned to GRCh37
- SV types: DEL/INS
- Minimum SV size: 50 bp
- Truth VCF: `real_data/HG002_chr21/truth_chr21/HG002_chr21_DELINS_50.vcf.gz`
- Tier1 BED: `real_data/HG002_chr21/truth_chr21/HG002_chr21_Tier1.bed`

## Truvari Benchmark Parameters

```bash
truvari bench \
  --includebed <Tier1 BED> \
  --passonly \
  --refdist 500 \
  --pctsize 0.5 \
  --sizemin 50 \
  --pctseq 0
```

## ReadGraphSV v0.3 Recommended Command

```bash
python run_readgraphsv_v2.py \
  --bam real_data/HG002_chr21/bam/HG002_chr21.bam \
  --model models/readgraph_gnn_v3.pt \
  --truth real_data/HG002_chr21/truth_chr21/HG002_chr21_DELINS_50.vcf.gz \
  --outdir runs/HG002_chr21_v3_final \
  --threshold 0.65 \
  --use_extra_candidates \
  --use_dedup \
  --dedup_window 500 \
  --dedup_min_size_sim 0.5
```

## Validation and Test Protocol

- Threshold source: chr19 validation selected `threshold=0.65`.
- Deduplication source: chr19 validation showed `window=500` and `min-size-sim=0.5` were effective.
- Held-out policy: chr21 final benchmark was used only as the held-out test and was not used for tuning.

## Final Comparison

| Tool | Precision | Recall | F1 | TP-comp | TP-base | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReadGraphSV v0.2 real-finetuned | 0.897143 | 0.887006 | 0.892045 | 157 | 157 | 18 | 20 |
| ReadGraphSV v0.3 trained | 0.898305 | 0.898305 | 0.898305 | 159 | 159 | 18 | 18 |
| ReadGraphSV v0.3 trained + dedup | 0.929412 | 0.892655 | 0.910663 | 158 | 158 | 12 | 19 |
| Sniffles2 | 0.923497 | 0.954802 | 0.938889 | 169 | 169 | 14 | 8 |
| cuteSV | 0.822967 | 0.971751 | 0.891192 | 172 | 172 | 37 | 5 |
| SVIM | 0.497024 | 0.943503 | 0.651072 | 167 | 167 | 169 | 10 |

## Conclusion

ReadGraphSV v0.3 + dedup achieved F1=0.9107 on HG002 chr21, outperforming cuteSV and SVIM in F1 and achieving higher precision than Sniffles2, while Sniffles2 retained the highest recall and overall F1.
