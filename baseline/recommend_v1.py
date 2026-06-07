"""MindTune v1 recommendation logic — frozen as a control baseline.

Wraps the existing v1 pipeline (2D arousal/valence, 4-strategy scoring)
so that v2 output can be compared against v1 output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.strategies import Phase, get_strategy  # noqa: E402
from src.scoring import select_tracks  # noqa: E402


# v1 emotion map (2D: arousal, valence)
V1_EMOTIONS = {
    "angry":      {"arousal": 0.8,  "valence": -0.7, "bpm_low": 130, "bpm_high": 170},
    "anxious":    {"arousal": 0.7,  "valence": -0.5, "bpm_low": 110, "bpm_high": 150},
    "sad":        {"arousal": -0.5, "valence": -0.7, "bpm_low": 50,  "bpm_high": 80},
    "melancholy": {"arousal": -0.3, "valence": -0.4, "bpm_low": 60,  "bpm_high": 90},
    "tired":      {"arousal": -0.8, "valence": -0.2, "bpm_low": 50,  "bpm_high": 70},
    "restless":   {"arousal": 0.4,  "valence": -0.2, "bpm_low": 100, "bpm_high": 130},
    "calm":       {"arousal": -0.5, "valence": 0.4,  "bpm_low": 60,  "bpm_high": 85},
    "peaceful":   {"arousal": -0.6, "valence": 0.6,  "bpm_low": 55,  "bpm_high": 80},
    "focused":    {"arousal": 0.2,  "valence": 0.3,  "bpm_low": 90,  "bpm_high": 120},
    "confident":  {"arousal": 0.5,  "valence": 0.7,  "bpm_low": 100, "bpm_high": 140},
    "excited":    {"arousal": 0.9,  "valence": 0.8,  "bpm_low": 120, "bpm_high": 160},
}


class _V1Emotion:
    def __init__(self, name: str, arousal: float, valence: float, bpm_low: int, bpm_high: int):
        self.name = name
        self.arousal = arousal
        self.valence = valence
        self.bpm_low = bpm_low
        self.bpm_high = bpm_high


def recommend_v1(
    source_label: str,
    target_label: str,
    tracks: list[dict],
    strategy_name: str = "Dynamic",
    tracks_per_phase: int = 3,
) -> dict:
    """Run v1 recommendation and return results as a dict.

    Returns:
        {
            "strategy": str,
            "source": str,
            "target": str,
            "phases": [{"label": str, "tracks": [dict]}],
            "all_track_ids": [str],
        }
    """
    src = V1_EMOTIONS[source_label.lower()]
    tgt = V1_EMOTIONS[target_label.lower()]

    source = _V1Emotion(source_label, **src)
    target = _V1Emotion(target_label, **tgt)

    strategy_fn = get_strategy(strategy_name)
    phases = strategy_fn(source, target)

    used_ids: set[str] = set()
    result_phases: list[dict] = []

    for phase in phases:
        selected = select_tracks(
            tracks, phase,
            count=tracks_per_phase,
            exclude_ids=used_ids,
        )
        for t in selected:
            used_ids.add(t.get("id", ""))
        result_phases.append({
            "label": phase.label,
            "tracks": selected,
        })

    return {
        "strategy": strategy_name,
        "source": source_label,
        "target": target_label,
        "phases": result_phases,
        "all_track_ids": list(used_ids),
    }


if __name__ == "__main__":
    # Quick test: load test data and run Angry→Calm
    test_path = _ROOT / "test_data" / "playlist_1498.json"
    if test_path.exists():
        with open(test_path) as f:
            tracks = json.load(f)
        result = recommend_v1("Angry", "Calm", tracks)
        print(json.dumps(result, indent=2, default=str)[:2000])
    else:
        print(f"Test data not found at {test_path}")
