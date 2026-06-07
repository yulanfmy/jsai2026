# LLM Feature-Extraction Performance Evaluation

Self-contained evaluation of the v2 feature extraction + correction pipeline
against Spotify ground truth (Zenodo dataset).

## Quick Start

```bash
cd <repo-root>
python eval/llm_feature_eval/run_eval.py
```

This will:
1. Load the 735 Zenodo-matched tracks from `cache/labeled_tracks.parquet`
2. Run 5-fold cross-validation (seed=42)
3. Compare three conditions: Raw LLM, Mean Predictor, v2 Corrected (Scheme 1+6)
4. Save scatter plots + error histograms to `eval/llm_feature_eval/outputs/`
5. Write the evaluation report to `eval/llm_feature_eval/REPORT.md`
6. Dump metrics JSON to `eval/llm_feature_eval/outputs/metrics.json`

## Prerequisites

- `cache/labeled_tracks.parquet` must exist (run "Build Feature Store" in the app first)
- Python packages: numpy, pandas, scipy, scikit-learn, lightgbm, matplotlib

## Isolation Guarantee

- **No production code was modified.** This evaluation only *imports* existing
  functions (`prepare_features`, `FEATURE_COLS`) for reproducibility.
- **No production model or cache was modified.** The correction model is
  re-trained inside the cross-validation loop (not saved to `models/`).
- **All outputs live under `eval/llm_feature_eval/outputs/`.** Nothing is
  written to `cache/`, `models/`, or any app directory.

## Output Files

```
eval/llm_feature_eval/
├── README.md          ← this file
├── REPORT.md          ← full evaluation report with tables and plots
├── run_eval.py        ← evaluation script
└── outputs/
    ├── metrics.json   ← machine-readable metrics
    ├── scatter_*.png  ← scatter plots (pred vs ground truth)
    └── hist_*.png     ← error histograms
```
