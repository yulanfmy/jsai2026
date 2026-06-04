"""Top-K candidate filter with arc direction and start-state protection (§6).

target_loss(track, stage, gamma=0.2):
  = position_distance_z + γ · arc_direction_mismatch
  - position_z: 3D Euclidean distance in z-scored (V,E,T) space so that
    no single axis dominates due to variance differences (§3.5).
  - arc_dir_mismatch: 1 - cos(track_arc_dir, expected_dir), range [0,2]

top_k_filter(tracks, stages, source_label, K=5):
  Per-stage filtering. Stage 1 start-state protection: for high-T starts,
  drop tracks whose arc raises T (arc_end_T > arc_start_T).
"""

from __future__ import annotations

import math

from src.config_loader import PARAMS
from src.planning.targets import StageTarget


def _cosine_similarity(a: tuple[float, float], b: tuple[float, float]) -> float:
    dot = a[0] * b[0] + a[1] * b[1]
    norm_a = math.sqrt(a[0] ** 2 + a[1] ** 2)
    norm_b = math.sqrt(b[0] ** 2 + b[1] ** 2)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def target_loss(
    track: dict,
    stage: StageTarget,
    gamma: float | None = None,
    zscore_params: dict | None = None,
) -> float:
    """Compute the target loss for a track at a given stage.

    Returns: position_distance_z + γ · arc_direction_mismatch

    When ``zscore_params`` is provided (recommended), the position distance
    is computed in z-scored space so no single axis dominates.
    """
    if gamma is None:
        gamma = PARAMS.gamma

    # Position distance in z-scored space (§3.5 / §6)
    if zscore_params is not None:
        from src.feature_store import standardize
        tV = float(track.get("zV", 0.0))
        tE = float(track.get("zE", 0.0))
        tT = float(track.get("zT", 0.0))
        sV, sE, sT = standardize(stage.V, stage.E, stage.T, zscore_params=zscore_params)
    else:
        tV = float(track.get("V", 0.0))
        tE = float(track.get("E", 0.0))
        tT = float(track.get("T", 0.0))
        sV, sE, sT = stage.V, stage.E, stage.T

    pos = math.sqrt((tV - sV) ** 2 + (tE - sE) ** 2 + (tT - sT) ** 2)

    # Arc direction mismatch (raw space — cosine is scale-invariant)
    raw_E = float(track.get("E", 0.0))
    raw_T = float(track.get("T", 0.0))
    track_dir = (
        float(track.get("arc_end_E", raw_E)) - float(track.get("arc_start_E", raw_E)),
        float(track.get("arc_end_T", raw_T)) - float(track.get("arc_start_T", raw_T)),
    )
    expected_dir = (stage.expected_dir_E, stage.expected_dir_T)
    cos_sim = _cosine_similarity(track_dir, expected_dir)
    arc_dir_mismatch = 1.0 - cos_sim  # range [0, 2]

    return pos + gamma * arc_dir_mismatch


def top_k_filter(
    all_tracks: list[dict],
    stages: list[StageTarget],
    source_label: str,
    K: int | None = None,
    gamma: float | None = None,
    zscore_params: dict | None = None,
) -> list[list[dict]]:
    """Filter to top-K candidates per stage.

    Args:
        all_tracks: full track library with features.
        stages: list of StageTarget (one per stage).
        source_label: source emotion label (for start-state protection).
        K: candidates per stage (default from params).
        gamma: arc-direction weight (default from params).
        zscore_params: persisted μ/σ for z-score distance (§3.5).

    Returns:
        List of K candidate lists, one per stage.
    """
    if K is None:
        K = PARAMS.K_default
    if gamma is None:
        gamma = PARAMS.gamma

    start_protection = set(PARAMS.start_protection)
    candidates_per_stage: list[list[dict]] = []

    for i, stage in enumerate(stages):
        pool = list(all_tracks)

        # Start-state protection: stage 1 only
        if i == 0 and source_label in start_protection:
            # Drop tracks whose arc raises T (we want T to decrease)
            filtered = [
                t for t in pool
                if (t.get("arc_end_T") or t.get("T") or 0) <= (t.get("arc_start_T") or t.get("T") or 0)
            ]
            # Relaxation: if too few candidates after filtering, keep at least K
            if len(filtered) >= K:
                pool = filtered

        # Score all tracks and take top-K
        scored = [(target_loss(t, stage, gamma, zscore_params), t) for t in pool]
        scored.sort(key=lambda x: x[0])
        top_k = [t for _, t in scored[:K]]

        candidates_per_stage.append(top_k)

    return candidates_per_stage
