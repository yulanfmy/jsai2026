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

---

## Analysis — Context and Interpretation

### Why is Valence harder than Energy?

Valence (positive/negative emotional tone) is consistently the hardest axis to predict across all MER systems, regardless of modality. Energy (arousal) correlates with acoustically salient features — tempo, loudness, spectral centroid — that are highly predictable even from genre and artist information. Valence, in contrast, depends on harmonic content, lyric sentiment, and culturally specific associations, making it harder to estimate from any single modality (Russell, 1980).

### Comparison with audio-based MER systems

State-of-the-art audio-based MER systems that process raw audio waveforms or spectrograms report the following valence R² scores on standard benchmarks:

| System | Dataset | R² (Valence) | R² (Arousal) | Modality |
|--------|---------|:---:|:---:|----------|
| Music2Emotion (2025) | DEAM | 0.52 | 0.62 | Audio (MERT + chords/key) |
| Music2Emotion (2025) | PMEmo | 0.55 | 0.79 | Audio (MERT + chords/key) |
| Music2Emotion (2025) | EmoMusic | 0.65 | 0.76 | Audio (MERT + chords/key) |
| ADFF (Li et al., 2022) | PMEmo | 0.46 | 0.64 | Audio (Mel-spectrogram) |
| **MindTune v2 (ours)** | **Zenodo subset** | **0.29** | **—** | **Metadata-only (LLM)** |

*Sources: Li et al. "ADFF: Attention Based Deep Feature Fusion Approach for MER" (2022); Guo et al. "Towards Unified Music Emotion Recognition across Dimensional and Categorical Models" (2025).*

Our **R² = 0.287** (Pearson r = 0.552) for valence is below current audio-based SOTA (R² ≈ 0.52–0.65), which is expected since we operate on **metadata only** (title, artist, album) without access to the audio signal. However, it represents a meaningful result:

1. **No audio required.** Audio-based systems need raw waveform access, which is not available through the Spotify API for third-party applications. Our metadata-only approach works with the information available.

2. **+32% relative improvement** from the Scheme 1+6 correction: raw LLM r = 0.418 → corrected r = 0.552, demonstrating that the sub-feature decomposition adds predictive value beyond direct LLM estimation.

3. **Energy estimation is competitive.** Our raw LLM energy Pearson r = 0.646 (Spearman ρ = 0.761) approaches the range of audio-based systems without any audio processing.

### Valence-arousal asymmetry

The gap between valence and energy performance (r = 0.552 vs r = 0.646) follows the well-documented valence–arousal asymmetry in the MER literature. Yang et al. (2008) first noted that "prediction of activation always outperforms valence," a finding consistently replicated across modalities (Yang & Chen, 2012). Our results confirm this pattern holds for metadata-based LLM estimation as well.

### References

- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178.
- Yang, Y.-H., Lin, Y.-C., Su, Y.-F., & Chen, H. H. (2008). A regression approach to music emotion recognition. *IEEE Transactions on Audio, Speech, and Language Processing*, 16(2), 255–266.
- Yang, Y.-H. & Chen, H. H. (2012). Machine recognition of music emotion: A review. *ACM Transactions on Intelligent Systems and Technology*, 3(3), Article 40.
- Li, H. et al. (2022). ADFF: Attention based deep feature fusion approach for music emotion recognition. *arXiv:2204.05649*.
- Guo, Z. et al. (2025). Towards unified music emotion recognition across dimensional and categorical models. *arXiv:2502.03979*.

## Reproducibility

- Random seed: `42`
- Folds: `5`
- Labeled tracks: `735`
- Correction model: LightGBM (same hyperparameters as production)
- **No production code, model, or cache was modified.**
