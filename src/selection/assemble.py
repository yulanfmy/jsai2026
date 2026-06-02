"""Deduplication and output assembly (§7.1, §7.2).

Post-process: swap duplicate tracks to next-best candidate.
Assemble: 6 ordered tracks + each track's stage + lead axis.
"""

from __future__ import annotations

from src.config_loader import PARAMS
from src.emotions import Emotion, get_emotion
from src.feature_store import get_all_for_user, get_zscore_params
from src.planning.axis_order import compute_axis_order
from src.planning.stage_alloc import allocate_stages
from src.planning.progress_matrix import build_progress_matrix
from src.planning.targets import StageTarget, compute_targets
from src.selection.candidates import target_loss, top_k_filter
from src.selection.viterbi import viterbi_select


def deduplicate(
    selected: list[dict],
    candidates: list[list[dict]],
    stages: list[StageTarget],
    gamma: float | None = None,
    zscore_params: dict | None = None,
) -> list[dict]:
    """Remove duplicates by swapping to next-best candidate.

    If a track appears at multiple stages, keep the one with lowest
    target_loss and swap the others to the next-best unused candidate.
    """
    if gamma is None:
        gamma = PARAMS.gamma

    result = list(selected)
    used_ids: set[str] = set()

    for i in range(len(result)):
        tid = result[i].get("id", "")
        if tid and tid in used_ids:
            # Find next-best unused candidate at this stage
            stage_candidates = candidates[i]
            sorted_cands = sorted(
                stage_candidates,
                key=lambda t: target_loss(t, stages[i], gamma, zscore_params),
            )
            replaced = False
            for cand in sorted_cands:
                cid = cand.get("id", "")
                if cid and cid not in used_ids:
                    result[i] = cand
                    used_ids.add(cid)
                    replaced = True
                    break
            if not replaced:
                # Keep the duplicate — no unused candidate available
                used_ids.add(tid)
        else:
            used_ids.add(tid)

    return result


def recommend_v2(
    source_label: str,
    target_label: str,
    user_id: str | None = None,
    K: int | None = None,
    N: int | None = None,
    alpha: float | None = None,
    lam: float | None = None,
    gamma: float | None = None,
) -> dict:
    """Full v2 recommendation pipeline.

    Input: two emotion labels → output N tracks with stage explanations.

    Returns dict with:
        source, target, axis_order, stage_alloc, stages, tracks,
        progress_matrix, all_track_ids
    """
    if K is None:
        K = PARAMS.K_default
    if N is None:
        N = PARAMS.N
    if alpha is None:
        alpha = PARAMS.alpha
    if lam is None:
        lam = PARAMS.lam
    if gamma is None:
        gamma = PARAMS.gamma

    source = get_emotion(source_label)
    target = get_emotion(target_label)

    # 1. Path planning (raw coordinate space)
    axis_order, delta = compute_axis_order(source, target, source_label)
    stage_alloc = allocate_stages(delta, axis_order, N)
    progress_matrix = build_progress_matrix(axis_order, stage_alloc, alpha, N)
    stages = compute_targets(source, target, delta, progress_matrix, axis_order, stage_alloc)

    # 2. Load tracks from feature store
    if user_id:
        all_tracks = get_all_for_user(user_id)
    else:
        from src.feature_store import get_all
        all_tracks = get_all()

    if not all_tracks:
        return {
            "source": source_label,
            "target": target_label,
            "error": "No tracks in feature store",
            "tracks": [],
        }

    # Load z-score params (§3.5) — same μ/σ for tracks AND targets
    zscore_params = get_zscore_params(user_id)

    # 3. Top-K candidate filter (distance in z-space)
    candidates = top_k_filter(all_tracks, stages, source_label, K, gamma, zscore_params)

    # 4. Viterbi DP
    selected = viterbi_select(candidates, stages, lam, gamma, zscore_params)

    # 5. Deduplication
    selected = deduplicate(selected, candidates, stages, gamma, zscore_params)

    # 6. Assemble output
    output_tracks: list[dict] = []
    for i, (track, stage) in enumerate(zip(selected, stages)):
        output_tracks.append({
            "track": track,
            "stage": i + 1,
            "lead_axis": stage.lead_axis,
            "target_V": stage.V,
            "target_E": stage.E,
            "target_T": stage.T,
            "explanation": stage.explanation,
        })

    return {
        "source": source_label,
        "target": target_label,
        "axis_order": axis_order,
        "stage_alloc": stage_alloc,
        "stages": [
            {"stage": s.stage, "V": s.V, "E": s.E, "T": s.T,
             "lead_axis": s.lead_axis, "dir_E": s.expected_dir_E, "dir_T": s.expected_dir_T}
            for s in stages
        ],
        "progress_matrix": progress_matrix,
        "tracks": output_tracks,
        "all_track_ids": [t["track"].get("id", "") for t in output_tracks],
    }
