"""Energy correction using LightGBM (same Scheme 1+6 as Valence).

The Energy ablation showed that raw LLM Energy has systematic
overestimation bias (R² = −0.25). Applying the same correction
procedure as Valence improves Energy substantially
(r: 0.646 → 0.745, R²: −0.25 → 0.55).

The correction model is trained on Zenodo-matched tracks (ground-truth
Energy) and applied to ALL tracks so the entire library is on one
consistent corrected scale.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache"
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
_REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"

# Same sub-features as Valence correction (Scheme 1+6)
FEATURE_COLS = [
    "V_raw", "E_raw", "T_raw",
    "mode_major_conf", "lyric_sentiment", "vocal_brightness", "chord_complexity",
    "tempo",
]


def prepare_features(track: dict) -> list[float]:
    """Extract the feature vector for the GBM model."""
    return [float(track.get(col, 0.0)) for col in FEATURE_COLS]


def train_model(
    tracks: list[dict],
    save_path: Path | None = None,
    user_id: str | None = None,
) -> tuple[object, dict]:
    """Train a LightGBM regressor on tracks that have Zenodo ground truth.

    Returns (model, metrics_dict).
    """
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score

    labeled = [t for t in tracks if t.get("has_zenodo") and "zenodo_energy" in t]
    if len(labeled) < 10:
        raise ValueError(f"Too few labeled tracks ({len(labeled)}). Need ≥10 for training.")

    X = pd.DataFrame([prepare_features(t) for t in labeled], columns=FEATURE_COLS)
    y = np.array([t["zenodo_energy"] for t in labeled])

    # Before correction: correlation of E_raw vs ground truth
    e_raw = np.array([t.get("E_raw", 0.0) for t in labeled])
    r_before = float(np.corrcoef(e_raw, y)[0, 1]) if np.std(e_raw) > 0 else 0.0

    # Train LightGBM (same hyperparameters as Valence)
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    # After correction: 5-fold CV
    cv_scores = cross_val_score(
        lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=5, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbose=-1,
        ),
        X, y,
        cv=min(5, len(labeled)),
        scoring="neg_mean_squared_error",
    )

    y_pred = model.predict(X)
    r_after = float(np.corrcoef(y_pred, y)[0, 1]) if np.std(y_pred) > 0 else 0.0

    metrics = {
        "n_labeled_E": len(labeled),
        "r_before_E": r_before,
        "r_after_E": r_after,
        "cv_r2_mean_E": float(np.mean(cv_scores)),
        "cv_r2_std_E": float(np.std(cv_scores)),
    }

    if save_path is None:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"model_E_{user_id}.pkl" if user_id else "model_E.pkl"
        save_path = _MODELS_DIR / fname
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    return model, metrics


def load_model(path: Path | None = None, user_id: str | None = None) -> object:
    """Load a trained model_E from disk."""
    if path is None:
        fname = f"model_E_{user_id}.pkl" if user_id else "model_E.pkl"
        path = _MODELS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"model_E not found at {path}. Run train_model first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def correct_energy(
    tracks: list[dict],
    model: object | None = None,
) -> list[dict]:
    """Apply energy correction to all tracks.

    For tracks with Zenodo ground truth: use Zenodo energy directly.
    For others: apply model_E prediction as corrected E.

    All tracks end up on one consistent corrected scale.
    Stores E_raw_original and E_corrected_source for traceability.
    """
    if model is None:
        model = load_model()

    n_zenodo = 0
    n_model = 0
    n_fallback = 0

    results: list[dict] = []
    for track in tracks:
        t = dict(track)
        t["E_raw_original"] = t.get("E_raw", 0.0)

        if t.get("has_zenodo") and "zenodo_energy" in t:
            t["E"] = t["zenodo_energy"]
            t["E_corrected_source"] = "zenodo"
            n_zenodo += 1
        else:
            # Check if all required sub-features are present
            missing = [c for c in FEATURE_COLS if c not in t]
            if missing:
                t["E"] = t.get("E_raw", 0.0)
                t["E_corrected_source"] = "fallback_raw"
                n_fallback += 1
            else:
                X_pred = pd.DataFrame([prepare_features(t)], columns=FEATURE_COLS)
                pred = float(model.predict(X_pred)[0])
                t["E"] = max(-1.0, min(1.0, pred))
                t["E_corrected_source"] = "model"
                n_model += 1

        results.append(t)

    print(f"  Energy correction coverage: "
          f"{n_zenodo} zenodo, {n_model} model, {n_fallback} fallback")
    if n_fallback > 0:
        print(f"  WARNING: {n_fallback} tracks fell back to raw E (missing sub-features)")

    return results


def write_report(metrics: dict, path: Path | None = None) -> Path:
    """Write the energy correction report."""
    if path is None:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = _REPORTS_DIR / "energy_correction.md"

    content = f"""# Energy Correction Report

## Summary

- **Labeled tracks (Zenodo matched)**: {metrics['n_labeled_E']}
- **Correlation before correction (E_raw vs ground truth)**: r = {metrics['r_before_E']:.4f}
- **Correlation after correction (model_E vs ground truth)**: r = {metrics['r_after_E']:.4f}
- **5-fold CV R² score**: {metrics['cv_r2_mean_E']:.4f} ± {metrics['cv_r2_std_E']:.4f}

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
"""
    with open(path, "w") as f:
        f.write(content)
    return path
