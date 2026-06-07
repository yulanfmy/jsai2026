"""3D emotion model with Valence (V), Energy arousal (E), Tension arousal (T).

Based on Schimmack & Reisenzein (2002): energy arousal and tension arousal
are driven by independent neurophysiological systems and should not be
collapsed into a single arousal axis.

Emotion coordinates are loaded from ``config/emotion_map.json`` via config_loader.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config_loader import EMOTION_MAP, PARAMS


@dataclass(frozen=True)
class Emotion:
    name: str
    V: float  # valence  [-1, +1]
    E: float  # energy arousal [-1, +1]
    T: float  # tension arousal [-1, +1]


START_PROTECTION: set[str] = set(PARAMS.start_protection)

EMOTIONS: dict[str, Emotion] = {
    key.lower(): Emotion(name=ec.name, V=ec.V, E=ec.E, T=ec.T)
    for key, ec in EMOTION_MAP.items()
}


def get_emotion(name: str) -> Emotion:
    key = name.lower().strip()
    if key not in EMOTIONS:
        raise ValueError(f"Unknown emotion: {name}. Choose from: {list(EMOTIONS.keys())}")
    return EMOTIONS[key]


def list_emotions() -> list[Emotion]:
    return list(EMOTIONS.values())


def emotion_from_vet(V: float, E: float, T: float, label: str = "Custom") -> Emotion:
    """Create a custom emotion from slider coordinates."""
    return Emotion(name=label, V=V, E=E, T=T)
