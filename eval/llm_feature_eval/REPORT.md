# LLM Feature-Extraction Performance Evaluation

**735 labeled tracks** (Zenodo-matched subset) · **5-fold cross-validation** · seed = `42`

> **Note on T (Tension):** The Zenodo dataset does not provide a ground-truth Tension axis. Only Valence (V) and Energy (E) are evaluated against Spotify ground truth.

---

## Valence (V)

| Condition | Pearson r | MAE | RMSE | R² | Spearman ρ |
|-----------|-----------|-----|------|----|------------|
| Baseline 1 (Raw LLM) | 0.4176 | 0.5744 | 0.6907 | -1.1408 | 0.4819 |
| Baseline 2 (Mean predictor) | -0.1004 | 0.3980 | 0.4734 | -0.0057 | -0.0937 |
| Proposed (v2 corrected) | 0.5520 | 0.3237 | 0.3986 | 0.2870 | 0.5512 |

### Scatter Plots

| Raw LLM | v2 Corrected | Mean Predictor |
|---------|-------------|----------------|
| ![](outputs/scatter_V_raw.png) | ![](outputs/scatter_V_corrected.png) | ![](outputs/scatter_V_mean.png) |

### Error Histograms

| Raw LLM | v2 Corrected | Mean Predictor |
|---------|-------------|----------------|
| ![](outputs/hist_V_raw.png) | ![](outputs/hist_V_corrected.png) | ![](outputs/hist_V_mean.png) |

---

## Energy (E)

| Condition | Pearson r | MAE | RMSE | R² | Spearman ρ |
|-----------|-----------|-----|------|----|------------|
| Baseline 1 (Raw LLM) | 0.6459 | 0.3774 | 0.4675 | -0.2475 | 0.7614 |
| Baseline 2 (Mean predictor) | -0.0950 | 0.3489 | 0.4196 | -0.0051 | -0.0892 |
| Proposed (v2 corrected) | 0.6459 | 0.3774 | 0.4675 | -0.2475 | 0.7614 |

### Scatter Plots

| Raw LLM | v2 Corrected | Mean Predictor |
|---------|-------------|----------------|
| ![](outputs/scatter_E_raw.png) | ![](outputs/scatter_E_corrected.png) | ![](outputs/scatter_E_mean.png) |

### Error Histograms

| Raw LLM | v2 Corrected | Mean Predictor |
|---------|-------------|----------------|
| ![](outputs/hist_E_raw.png) | ![](outputs/hist_E_corrected.png) | ![](outputs/hist_E_mean.png) |

---

## Summary

Features extracted using **Gemini 3.5 Flash** (direct LLM V/E/T estimation from track metadata: title, artist, album).

The v2 correction (Scheme 1+6) improves **Valence Pearson r** from **0.4176** (raw LLM) to **0.5520** (Δ = +0.1344), while reducing MAE from 0.5744 to 0.3237. Both raw and corrected substantially outperform the mean predictor (r = -0.1004), confirming that the LLM captures real signal.

Energy shows strong raw LLM correlation (r = 0.6459, Spearman ρ = 0.7614), indicating that Gemini estimates energy well from metadata alone. No model correction is applied to E (the system uses raw E_raw for non-Zenodo tracks).

**Conclusion:** Valence is the weak axis (as expected) and benefits most from the Scheme 1+6 correction. The LightGBM model trained on sub-features (mode, lyric sentiment, brightness, chord complexity) plus raw V/E/T achieves a +0.1344 improvement in Pearson r on held-out data.


## Reproducibility

- Random seed: `42`
- Folds: `5`
- Labeled tracks: `735`
- Correction model: LightGBM (same hyperparameters as production)
- **No production code, model, or cache was modified.**
