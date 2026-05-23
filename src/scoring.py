"""Track scoring algorithm for playlist generation.

Implements the weighted scoring system from the paper:
  - energy fit:          weight 3
  - happiness fit:       weight 3
  - BPM fit:             weight 2
  - vibe keyword match:  weight 1
  - instrumentalness:    weight 2 (filter for lyrics preference)
"""

from src.strategies import Phase

WEIGHT_ENERGY = 3
WEIGHT_HAPPINESS = 3
WEIGHT_BPM = 2
WEIGHT_VIBE = 1
WEIGHT_INSTRUMENTAL = 2

ENERGY_TOLERANCE = 0.25
HAPPINESS_TOLERANCE = 0.25
BPM_TOLERANCE = 20


def _in_range(value: float, target: float, tolerance: float) -> bool:
    return abs(value - target) <= tolerance


def _range_score(value: float, target: float, tolerance: float) -> float:
    diff = abs(value - target)
    if diff <= tolerance:
        return 1.0
    return max(0.0, 1.0 - (diff - tolerance) / tolerance)


def _bpm_in_range(bpm: int, low: int, high: int) -> float:
    if low <= bpm <= high:
        return 1.0
    dist = min(abs(bpm - low), abs(bpm - high))
    return max(0.0, 1.0 - dist / BPM_TOLERANCE)


def _arousal_to_energy(arousal: float) -> float:
    """Map arousal [-1, 1] to energy [0, 1]."""
    return (arousal + 1.0) / 2.0


def _valence_to_happiness(valence: float) -> float:
    """Map valence [-1, 1] to happiness [0, 1]."""
    return (valence + 1.0) / 2.0


def score_track(
    track: dict,
    phase: Phase,
    vibe_keywords: list[str] | None = None,
    prefer_vocals: bool | None = None,
) -> float | None:
    """Score a track against a phase's requirements.

    Returns None if the track should be excluded (both energy and happiness
    far out of range), otherwise returns a normalized weighted score.
    """
    target_energy = _arousal_to_energy(phase.arousal)
    target_happiness = _valence_to_happiness(phase.valence)

    energy = track.get("energy", 0.5)
    happiness = track.get("happiness", 0.5)
    bpm = track.get("bpm", 100)
    instrumentalness = track.get("instrumentalness", 0.5)
    track_vibe = track.get("vibe", "").lower()

    energy_ok = _in_range(energy, target_energy, ENERGY_TOLERANCE)
    happiness_ok = _in_range(happiness, target_happiness, HAPPINESS_TOLERANCE)

    if not energy_ok and not happiness_ok:
        energy_close = abs(energy - target_energy) < ENERGY_TOLERANCE * 2
        happiness_close = abs(happiness - target_happiness) < HAPPINESS_TOLERANCE * 2
        if not energy_close and not happiness_close:
            return None

    energy_score = _range_score(energy, target_energy, ENERGY_TOLERANCE)
    happiness_score = _range_score(happiness, target_happiness, HAPPINESS_TOLERANCE)
    bpm_score = _bpm_in_range(bpm, phase.bpm_low, phase.bpm_high)

    vibe_score = 0.0
    if vibe_keywords:
        for kw in vibe_keywords:
            if kw.lower() in track_vibe:
                vibe_score = 1.0
                break

    instrumental_score = 0.0
    if prefer_vocals is None:
        instrumental_score = 0.5
    elif prefer_vocals:
        instrumental_score = 1.0 - instrumentalness
    else:
        instrumental_score = instrumentalness

    total = (
        WEIGHT_ENERGY * energy_score
        + WEIGHT_HAPPINESS * happiness_score
        + WEIGHT_BPM * bpm_score
        + WEIGHT_VIBE * vibe_score
        + WEIGHT_INSTRUMENTAL * instrumental_score
    )
    max_possible = (
        WEIGHT_ENERGY + WEIGHT_HAPPINESS + WEIGHT_BPM + WEIGHT_VIBE + WEIGHT_INSTRUMENTAL
    )
    return total / max_possible


def select_tracks(
    tracks: list[dict],
    phase: Phase,
    count: int = 3,
    exclude_ids: set[str] | None = None,
    vibe_keywords: list[str] | None = None,
    prefer_vocals: bool | None = None,
) -> list[dict]:
    """Select the best tracks for a given phase.

    Returns up to ``count`` tracks, sorted by score descending.
    Tracks in ``exclude_ids`` are skipped to prevent cross-phase duplicates.
    """
    if exclude_ids is None:
        exclude_ids = set()

    scored: list[tuple[float, dict]] = []
    for track in tracks:
        tid = track.get("id", track.get("title", ""))
        if tid in exclude_ids:
            continue
        s = score_track(track, phase, vibe_keywords, prefer_vocals)
        if s is not None:
            scored.append((s, track))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:count]]
