# LLM Feature-Extraction Performance Evaluation

**735 labeled tracks** (Zenodo-matched subset) · **5-fold cross-validation** · seed = `42`

> **Note on T (Tension):** The Zenodo dataset does not provide a ground-truth Tension axis. Only Valence (V) and Energy (E) are evaluated against Spotify ground truth.

---

## Valence (V)

| Condition | Pearson r | MAE | RMSE | R² | Spearman ρ |
|-----------|-----------|-----|------|----|------------|
| Baseline 1 (Raw LLM) | -0.0474 | 0.5918 | 0.7255 | -1.3622 | -0.0479 |
| Baseline 2 (Mean predictor) | -0.1004 | 0.3980 | 0.4734 | -0.0057 | -0.0937 |
| Proposed (v2 corrected) | -0.0165 | 0.4347 | 0.5297 | -0.2593 | -0.0084 |

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
| Baseline 1 (Raw LLM) | 0.0244 | 0.5536 | 0.6797 | -1.6372 | 0.0241 |
| Baseline 2 (Mean predictor) | -0.0950 | 0.3489 | 0.4196 | -0.0051 | -0.0892 |
| Proposed (v2 corrected) | 0.0244 | 0.5536 | 0.6797 | -1.6372 | 0.0241 |

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

**Important context:** The current features are **bootstrapped from v1** (mathematical formulas: `V_raw = 2*happiness - 1`, `E_raw = 2*energy - 1`), not from direct LLM extraction of V/E/T. The v1 features (energy, happiness) are OpenAI estimates of subjective qualities, which do not map closely to Spotify's audio-derived valence and energy.

All three conditions show near-zero Pearson r for both axes, indicating that the bootstrapped features carry minimal linear signal relative to Spotify ground truth. For Valence: raw r = -0.0474, corrected r = -0.0165 (Δ = +0.0310). The v2 correction reduces MAE from 0.5918 to 0.4347, showing that the GBM model partially compensates for the systematic bias in the raw features.

For Energy: raw r = 0.0244. No model correction is applied to E (the system uses raw E_raw for non-Zenodo tracks).

**Conclusion:** The bootstrapped v1→v2 conversion does not produce features that correlate with Spotify audio features. To achieve meaningful correlation, direct LLM extraction (`llm_extract.py` with a capable model like Gemini) should be used instead of the v1 bootstrap.


## Reproducibility

- Random seed: `42`
- Folds: `5`
- Labeled tracks: `735`
- Correction model: LightGBM (same hyperparameters as production)
- **No production code, model, or cache was modified.**
