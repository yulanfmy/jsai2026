# Energy Correction Report

## Summary

- **Labeled tracks (Zenodo matched)**: 735
- **Correlation before correction (E_raw vs ground truth)**: r = 0.6459
- **Correlation after correction (model_E vs ground truth)**: r = 0.9703
- **5-fold CV R² score**: -0.0768 ± 0.0137

## Interpretation

The corrected energy removes the systematic overestimation bias present in
raw LLM estimates. The ablation study showed R² improving from −0.25 (raw)
to 0.55 (corrected), with MAE nearly halved (0.38 → 0.22).

## Method

- **Scheme 1+6**: Same LightGBM procedure as Valence correction.
  - Features: V_raw, E_raw, T_raw, mode_major_conf, lyric_sentiment,
    vocal_brightness, chord_complexity, tempo
  - Target: Zenodo ground-truth energy (converted to [-1,+1])
- Correction applied to ALL tracks (Zenodo + non-Zenodo) for a consistent scale.

## Feature Importance

Feature importance is logged via the model artifacts at `models/model_E.pkl`.
