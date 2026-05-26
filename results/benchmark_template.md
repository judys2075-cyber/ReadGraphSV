# Benchmark Results Template

Use this template to report raw candidate performance and GNN-filtered
performance for a held-out dataset.

## Dataset

- Dataset:
- Read type:
- Coverage:
- Reference:
- Truth source:
- Truth variants:
- Supported SV types in this run:
- Candidate extraction parameters:
  - `min_size`:
  - `window`:
  - `min_support`:
- GNN checkpoint:
- Decision threshold:

## Summary Metrics

| Method | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|
| Raw CIGAR candidates | TBD | TBD | TBD | N/A |
| ReadGraphSV GNN-filtered | TBD | TBD | TBD | TBD |

## Confusion Matrix

| Method | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| Raw CIGAR candidates | TBD | TBD | TBD | TBD |
| ReadGraphSV GNN-filtered | TBD | TBD | TBD | TBD |

## Filtering Effect

- FP reduction:
- FP reduction rate:
- TP loss:
- TP loss rate:

## Notes

- Raw candidates are evaluated by treating every candidate as positive.
- GNN-filtered calls are evaluated after applying the selected probability
  threshold.
- When reporting final test metrics, select model checkpoints and thresholds
  using only the training/validation data.
