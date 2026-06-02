"""Four emotion-transition strategies based on the ISO principle.

Each strategy divides the transition from current to target emotion into
3 phases, producing intermediate (arousal, valence) waypoints that guide
track selection.
"""

from dataclasses import dataclass
from typing import Callable

from src.emotions import Emotion


@dataclass
class Phase:
    label: str
    arousal: float
    valence: float
    bpm_low: int
    bpm_high: int


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _bpm_range(arousal: float, valence: float) -> tuple[int, int]:
    """Estimate a BPM range from arousal/valence coordinates."""
    base = 70 + int(arousal * 50)
    spread = 20
    low = max(40, base - spread)
    high = min(200, base + spread)
    return low, high


def arousal_first(current: Emotion, target: Emotion) -> list[Phase]:
    """Adjust arousal (E) first, then shift valence (V)."""
    mid_arousal = target.E
    mid_valence = current.V
    bpm1 = _bpm_range(_lerp(current.E, mid_arousal, 0.5), current.V)
    bpm2 = _bpm_range(mid_arousal, _lerp(current.V, target.V, 0.5))
    bpm3 = _bpm_range(target.E, target.V)
    return [
        Phase(
            "Adjust Arousal",
            _lerp(current.E, mid_arousal, 0.5),
            current.V,
            bpm1[0],
            bpm1[1],
        ),
        Phase(
            "Shift Valence",
            mid_arousal,
            _lerp(mid_valence, target.V, 0.5),
            bpm2[0],
            bpm2[1],
        ),
        Phase("Reach Target", target.E, target.V, bpm3[0], bpm3[1]),
    ]


def valence_first(current: Emotion, target: Emotion) -> list[Phase]:
    """Shift valence (V) first, then adjust arousal (E)."""
    mid_arousal = current.E
    mid_valence = target.V
    bpm1 = _bpm_range(current.E, _lerp(current.V, mid_valence, 0.5))
    bpm2 = _bpm_range(_lerp(current.E, target.E, 0.5), mid_valence)
    bpm3 = _bpm_range(target.E, target.V)
    return [
        Phase(
            "Shift Valence",
            current.E,
            _lerp(current.V, mid_valence, 0.5),
            bpm1[0],
            bpm1[1],
        ),
        Phase(
            "Adjust Arousal",
            _lerp(mid_arousal, target.E, 0.5),
            mid_valence,
            bpm2[0],
            bpm2[1],
        ),
        Phase("Reach Target", target.E, target.V, bpm3[0], bpm3[1]),
    ]


def linear(current: Emotion, target: Emotion) -> list[Phase]:
    """Change arousal (E) and valence (V) simultaneously at equal pace."""
    phases = []
    for i, label in enumerate(["Begin Transition", "Mid Transition", "Reach Target"], 1):
        frac = i / 3.0
        a = _lerp(current.E, target.E, frac)
        v = _lerp(current.V, target.V, frac)
        bpm = _bpm_range(a, v)
        phases.append(Phase(label, a, v, bpm[0], bpm[1]))
    return phases


def dynamic(current: Emotion, target: Emotion) -> list[Phase]:
    """Automatically prioritize the dimension with the larger gap."""
    arousal_gap = abs(target.E - current.E)
    valence_gap = abs(target.V - current.V)
    if arousal_gap >= valence_gap:
        return arousal_first(current, target)
    return valence_first(current, target)


STRATEGIES: dict[str, Callable[[Emotion, Emotion], list[Phase]]] = {
    "Arousal First": arousal_first,
    "Valence First": valence_first,
    "Linear": linear,
    "Dynamic": dynamic,
}


def get_strategy(name: str) -> Callable[[Emotion, Emotion], list[Phase]]:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Choose from: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]


def get_strategy_description(name: str) -> str:
    descriptions = {
        "Arousal First": (
            "Adjusts energy level first, then shifts mood. "
            "Recommended for high-arousal negative states (anxiety, anger)."
        ),
        "Valence First": (
            "Shifts mood first, then adjusts energy. "
            "Recommended for low-arousal negative states (sadness, fatigue)."
        ),
        "Linear": (
            "Changes arousal and valence simultaneously at an equal pace. "
            "Best for moderate transitions or when current and target are close."
        ),
        "Dynamic": (
            "Automatically prioritizes the dimension with the larger gap. "
            "Generally the most effective strategy for smooth transitions."
        ),
    }
    return descriptions.get(name, "")
