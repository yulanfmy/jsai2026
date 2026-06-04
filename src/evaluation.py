"""Evaluation-only helpers: Spotify Autoplay baseline + blind A/B/C assignment.

This module is ONLY active when EVAL_MODE is True.
When EVAL_MODE is False, none of these functions are called.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from src.spotify import get_recommendations, get_top_tracks

_LOG_DIR = Path("cache/eval_logs")


def spotify_autoplay_baseline(
    access_token: str,
    N: int = 6,
) -> dict:
    """Generate a Spotify Autoplay baseline playlist (emotion-agnostic).

    Seed: user's top tracks (taste-based, no V/E/T input).
    Tries the Recommendations API first; falls back to top tracks if unavailable.

    Returns the same output shape as recommend_v2 for uniform presentation.
    """
    # Get user's top tracks as seeds
    top_tracks = get_top_tracks(access_token, limit=20, time_range="medium_term")
    if not top_tracks:
        top_tracks = get_top_tracks(access_token, limit=20, time_range="long_term")

    if not top_tracks:
        return {
            "source": "autoplay",
            "target": "autoplay",
            "method": "spotify_autoplay",
            "error": "No top tracks available",
            "tracks": [],
        }

    seed_ids = [t["id"] for t in top_tracks[:5]]

    # Try Recommendations API (may be deprecated for new apps)
    reco_tracks = get_recommendations(access_token, seed_ids, limit=N)

    if reco_tracks:
        method_detail = "spotify_recommendations_api"
        result_tracks = reco_tracks[:N]
    else:
        # Fallback: use top tracks themselves (shuffled)
        method_detail = "top_tracks_fallback"
        shuffled = list(top_tracks)
        random.shuffle(shuffled)
        result_tracks = shuffled[:N]

    # Assemble in recommend_v2-compatible shape
    output_tracks = []
    for i, track in enumerate(result_tracks):
        output_tracks.append({
            "track": track,
            "stage": i + 1,
            "lead_axis": "-",
            "target_V": 0.0,
            "target_E": 0.0,
            "target_T": 0.0,
            "explanation": f"Autoplay track {i + 1}",
        })

    return {
        "source": "autoplay",
        "target": "autoplay",
        "method": "spotify_autoplay",
        "method_detail": method_detail,
        "seed_ids": seed_ids,
        "tracks": output_tracks,
    }


def assign_blind_labels(methods: list[str]) -> dict[str, str]:
    """Randomly assign A/B/C labels to method names.

    Args:
        methods: list of method names, e.g. ["dynamic", "linear", "autoplay"]

    Returns:
        {"A": "dynamic", "B": "autoplay", "C": "linear"} (randomized)
    """
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:len(methods)]
    shuffled = list(methods)
    random.shuffle(shuffled)
    return {label: method for label, method in zip(labels, shuffled)}


def log_eval_session(
    user_id: str,
    mapping: dict[str, str],
    results: dict[str, dict],
    ratings: dict[str, object] | None = None,
) -> Path:
    """Write per-session evaluation log (JSON).

    Records: method<->letter mapping, seed, track IDs, ratings.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    log_path = _LOG_DIR / f"eval_{user_id}_{ts}.json"

    log_data = {
        "timestamp": ts,
        "user_id": user_id,
        "mapping": mapping,  # {"A": "dynamic", "B": "linear", ...}
        "results": {},
        "ratings": ratings or {},
    }

    for label, method in mapping.items():
        res = results.get(method, {})
        log_data["results"][label] = {
            "method": method,
            "track_ids": [
                item["track"].get("id", "") for item in res.get("tracks", [])
            ],
            "seed_ids": res.get("seed_ids", []),
            "method_detail": res.get("method_detail", ""),
        }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    return log_path
