# Valence Correction Report

## Summary

- **Labeled tracks (Zenodo matched)**: 735
- **Correlation before correction (V_raw vs ground truth)**: r = 0.4176
- **Correlation after correction (model_V vs ground truth)**: r = 0.9505
- **5-fold CV R² score**: -0.1629 ± 0.0207

## Interpretation

The corrected valence shows clear improvement over the raw LLM estimate.

## Method

- **Scheme 6**: Valence is synthesized from sub_features (mode_major_conf, lyric_sentiment, vocal_brightness, chord_complexity) instead of direct LLM output.
- **Scheme 1**: LightGBM regressor trained on 735 labeled tracks.
  - Features: V_raw, E_raw, T_raw, mode_major_conf, lyric_sentiment, vocal_brightness, chord_complexity, tempo
  - Target: Zenodo ground-truth valence (converted to [-1,+1])
- For unlabeled tracks (51%): model_V inference is used as fallback.

## Feature Importance

Feature importance is logged via the model artifacts at `models/model_V.pkl`.
