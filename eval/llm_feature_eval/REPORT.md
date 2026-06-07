# LLM Feature-Extraction Performance Evaluation

**4987 labeled tracks** (Zenodo-matched subset) · **5-fold cross-validation** · seed = `42`

> **Note on T (Tension):** The Zenodo dataset does not provide a ground-truth Tension axis. Only Valence (V) and Energy (E) are evaluated against Spotify ground truth.

---

## Valence (V)

| Condition | Pearson r | MAE | RMSE | R² | Spearman ρ |
|-----------|-----------|-----|------|----|------------|
| Baseline 1 (Raw LLM) | 0.3641 | 0.5585 | 0.7012 | -0.5056 | 0.4030 |
| Baseline 2 (Mean predictor) | -0.0305 | 0.4963 | 0.5716 | -0.0005 | -0.0293 |
| Proposed (v2 corrected) | 0.5499 | 0.3951 | 0.4775 | 0.3019 | 0.5487 |

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
| Baseline 1 (Raw LLM) | 0.6267 | 0.4968 | 0.6315 | -0.2088 | 0.6821 |
| Baseline 2 (Mean predictor) | -0.0300 | 0.4975 | 0.5745 | -0.0005 | -0.0273 |
| Proposed (v2 corrected) | 0.7845 | 0.2753 | 0.3563 | 0.6152 | 0.7808 |

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

The v2 correction (Scheme 1+6) improves **Valence Pearson r** from **0.3641** (raw LLM) to **0.5499** (Δ = +0.1858), while reducing MAE from 0.5585 to 0.3951. Both raw and corrected substantially outperform the mean predictor (r = -0.0305), confirming that the LLM captures real signal.

For **Energy**, the v2 correction improves Pearson r from **0.6267** (raw LLM) to **0.7845** (Δ = +0.1578).

**Conclusion:** Both Valence and Energy benefit from the Scheme 1+6 correction. The LightGBM model trained on sub-features (mode, lyric sentiment, brightness, chord complexity, tempo feel, dynamic range, rhythmic density, distortion level) plus raw V/E/T achieves meaningful improvement on held-out data for both axes.


## Reproducibility

- Random seed: `42`
- Folds: `5`
- Labeled tracks: `4987`
- Correction model: LightGBM (same hyperparameters as production)
- **No production code, model, or cache was modified.**
