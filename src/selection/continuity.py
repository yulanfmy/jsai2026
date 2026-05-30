"""Musical continuity functions (§5 of spec).

DJ-heuristic functions for evaluating adjacent-track transitions:
  - key_compat: Camelot wheel key compatibility (1.0/0.9/0.8/0.3)
  - tempo_compat: BPM compatibility (±12% limit → [0,1])
  - arc_join: emotional arc join distance (Euclidean in E,T)
"""

from __future__ import annotations

import math

# Camelot wheel mapping: (key 0-11, mode 0/1) → (number 1-12, letter A/B)
# key: 0=C, 1=C#, 2=D, ..., 11=B
# mode: 0=minor(A), 1=major(B)
_CAMELOT: dict[tuple[int, int], tuple[int, str]] = {
    (0, 1): (8, "B"),   # C major
    (1, 1): (3, "B"),   # C# major
    (2, 1): (10, "B"),  # D major
    (3, 1): (5, "B"),   # Eb major
    (4, 1): (12, "B"),  # E major
    (5, 1): (7, "B"),   # F major
    (6, 1): (2, "B"),   # F# major
    (7, 1): (9, "B"),   # G major
    (8, 1): (4, "B"),   # Ab major
    (9, 1): (11, "B"),  # A major
    (10, 1): (6, "B"),  # Bb major
    (11, 1): (1, "B"),  # B major
    (0, 0): (5, "A"),   # C minor
    (1, 0): (12, "A"),  # C# minor
    (2, 0): (7, "A"),   # D minor
    (3, 0): (2, "A"),   # Eb minor
    (4, 0): (9, "A"),   # E minor
    (5, 0): (4, "A"),   # F minor
    (6, 0): (11, "A"),  # F# minor
    (7, 0): (6, "A"),   # G minor
    (8, 0): (1, "A"),   # Ab minor
    (9, 0): (8, "A"),   # A minor
    (10, 0): (3, "A"),  # Bb minor
    (11, 0): (10, "A"), # B minor
}


def _camelot(key: int, mode: int) -> tuple[int, str]:
    return _CAMELOT.get((key % 12, mode), (1, "A"))


def key_compat(a_key: int, a_mode: int, b_key: int, b_mode: int) -> float:
    """Camelot key compatibility score.

    Returns:
        1.0  same key
        0.9  adjacent on Camelot wheel (±1 number, same letter)
        0.8  relative major/minor (same number, different letter)
        0.3  everything else
    """
    a_num, a_let = _camelot(a_key, a_mode)
    b_num, b_let = _camelot(b_key, b_mode)

    if a_num == b_num and a_let == b_let:
        return 1.0

    # Adjacent: same letter, ±1 (mod 12)
    if a_let == b_let:
        diff = abs(a_num - b_num)
        if diff == 1 or diff == 11:  # wrap around 12→1
            return 0.9

    # Relative major/minor: same number, different letter
    if a_num == b_num and a_let != b_let:
        return 0.8

    return 0.3


def tempo_compat(a_tempo: float, b_tempo: float) -> float:
    """Tempo compatibility based on ±12% limit.

    Returns [0, 1]: 1.0 for same BPM, 0 for ≥12% difference.
    """
    if a_tempo <= 0 or b_tempo <= 0:
        return 0.5  # unknown tempo
    ratio = abs(a_tempo - b_tempo) / max(a_tempo, b_tempo)
    limit = 0.12
    if ratio >= limit:
        return 0.0
    return 1.0 - ratio / limit


def arc_join(
    a_end_E: float, a_end_T: float,
    b_start_E: float, b_start_T: float,
) -> float:
    """Arc join distance — Euclidean distance between track A's end and track B's start.

    Lower is better (smoother emotional handoff).
    """
    return math.sqrt((a_end_E - b_start_E) ** 2 + (a_end_T - b_start_T) ** 2)
