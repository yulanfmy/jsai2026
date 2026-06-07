"""Bootstrap v2 features from existing v1 track data.

Maps v1 fields (energy [0,1], happiness [0,1], bpm, vibe, instrumentalness)
to the v2 schema (V_raw, E_raw, T_raw, arc, sub_features, key, mode, tempo).

This is a stopgap: the full LLM extraction (llm_extract.py) produces
higher-quality estimates. Run the LLM extraction when an API key is available.
"""

from __future__ import annotations

import math
import random


def bootstrap_track(track: dict) -> dict:
    """Convert a v1 track dict to v2 feature schema."""
    result = dict(track)

    # Map v1 energy [0,1] → v2 E [-1,+1]
    energy_01 = track.get("energy", 0.5)
    result["E_raw"] = 2.0 * energy_01 - 1.0
    result["E"] = result["E_raw"]

    # Map v1 happiness [0,1] → v2 V [-1,+1]
    happiness_01 = track.get("happiness", 0.5)
    result["V_raw"] = 2.0 * happiness_01 - 1.0
    result["V"] = result["V_raw"]

    # Derive T (tension arousal) from energy + vibe heuristics
    # High energy + negative vibes → high tension; low energy → low tension
    vibe = track.get("vibe", "").lower()
    tension_vibes = {"intense", "aggressive", "dark", "angry", "anxious", "chaotic",
                     "heavy", "powerful", "epic", "dramatic", "fierce", "raw"}
    calm_vibes = {"chill", "calm", "peaceful", "dreamy", "soft", "gentle",
                  "serene", "tranquil", "mellow", "ambient", "relaxed"}

    t_base = (energy_01 - 0.5) * 0.6  # energy contributes to tension
    if vibe in tension_vibes:
        t_base += 0.3
    elif vibe in calm_vibes:
        t_base -= 0.3
    # Negative valence contributes to tension
    t_base += max(0, -result["V_raw"]) * 0.2
    result["T_raw"] = max(-1.0, min(1.0, t_base))
    result["T"] = result["T_raw"]

    # Arc vectors: slight variation from the midpoint (simple heuristic)
    # Tracks generally have a subtle arc from start to end
    arc_delta_E = random.gauss(0, 0.1)
    arc_delta_T = random.gauss(0, 0.1)
    result["arc_start_E"] = max(-1.0, min(1.0, result["E"] - arc_delta_E / 2))
    result["arc_start_T"] = max(-1.0, min(1.0, result["T"] - arc_delta_T / 2))
    result["arc_end_E"] = max(-1.0, min(1.0, result["E"] + arc_delta_E / 2))
    result["arc_end_T"] = max(-1.0, min(1.0, result["T"] + arc_delta_T / 2))

    # Sub-features (decomposition of valence)
    result["mode_major_conf"] = max(0.0, min(1.0, (happiness_01 + 0.2)))
    result["lyric_sentiment"] = result["V_raw"]
    result["vocal_brightness"] = max(0.0, min(1.0, energy_01 * 0.8 + 0.1))
    result["chord_complexity"] = max(0.0, min(1.0, 0.5 + (1 - happiness_01) * 0.3))

    # Key and mode — derive from BPM and vibe heuristically
    bpm = track.get("bpm", 120)
    result["tempo"] = float(bpm)
    # Simple hash-based key assignment (deterministic per track)
    tid = track.get("id", track.get("title", ""))
    hash_val = hash(tid)
    result["key"] = abs(hash_val) % 12
    result["mode"] = 1 if happiness_01 > 0.5 else 0

    # Lyrics
    instrumentalness = track.get("instrumentalness", 0.0)
    result["lyrics_present"] = instrumentalness < 0.5

    result["vibe"] = track.get("vibe", "")

    return result


def bootstrap_all(tracks: list[dict]) -> list[dict]:
    """Bootstrap v2 features for all tracks. Deterministic with seed."""
    random.seed(42)
    return [bootstrap_track(t) for t in tracks]
