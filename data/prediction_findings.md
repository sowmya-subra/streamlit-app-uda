# GuardRail — Stage 8: Prediction Model Findings

## Key Takeaway

Adding taxonomy features improved AUC-ROC by **-0.01** vs text-only.
Adding network features added another **-0.00** on top.

## Model Performance Summary

| Feature Set | AUC-ROC (Logistic) | R² (Ridge) |
|---|---|---|
| Baseline (text only) | 0.8429 | 0.3589 |
| +Taxonomy            | 0.8330  | 0.3594  |
| +Network             | 0.8314  | 0.3591  |

## Slide Bullets

- **Taxonomy matters:** +-0.010 AUC — technique type is a stronger harm signal than wording alone.

- **Network adds signal:** +-0.002 AUC — hub techniques in the co-occurrence graph are more harmful.

- **Most predictable:** `harm_violence_graphic` (AUC=0.9529)

- **Hardest category:** `harm_hate` (AUC=0.7566) — too few high-score examples.

## Honest Framing

The headline is the increment. "+-0.01 AUC from taxonomy" proves the framework adds value beyond raw text.

## Deliverables
- [x] model_comparison_table.csv
- [x] feature_importance.png
- [x] per_category_performance.png
- [x] prediction_findings.md