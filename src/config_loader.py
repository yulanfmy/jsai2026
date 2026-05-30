"""Centralized config loader for MindTune v2.

Reads parameters from config/params.yaml and config/emotion_map.json.
All code accesses parameters through this module — no hard-coded numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_PARAMS_PATH = _ROOT / "config" / "params.yaml"
_EMOTION_MAP_PATH = _ROOT / "config" / "emotion_map.json"


@dataclass(frozen=True)
class Params:
    """All fixed parameters from params.yaml."""

    N: int = 6
    K_default: int = 5
    alpha: float = 0.6
    lam: float = 0.4         # 'lambda' is a Python keyword, so we use 'lam'
    gamma: float = 0.2
    w1: float = 0.4
    w2: float = 0.4
    w3: float = 0.2
    start_protection: tuple[str, ...] = ("Angry", "Anxious", "Fear")


def load_params(path: Path | None = None) -> Params:
    """Load parameters from params.yaml."""
    if path is None:
        path = _PARAMS_PATH
    if not path.exists():
        return Params()

    with open(path) as f:
        raw = yaml.safe_load(f)

    return Params(
        N=raw.get("N", 6),
        K_default=raw.get("K_default", 5),
        alpha=raw.get("alpha", 0.6),
        lam=raw.get("lambda", 0.4),
        gamma=raw.get("gamma", 0.2),
        w1=raw.get("w1", 0.4),
        w2=raw.get("w2", 0.4),
        w3=raw.get("w3", 0.2),
        start_protection=tuple(raw.get("start_protection", ["Angry", "Anxious", "Fear"])),
    )


@dataclass(frozen=True)
class EmotionCoord:
    """A single emotion's (V, E, T) coordinates."""

    name: str
    V: float
    E: float
    T: float


def load_emotion_map(path: Path | None = None) -> dict[str, EmotionCoord]:
    """Load the emotion map from emotion_map.json."""
    if path is None:
        path = _EMOTION_MAP_PATH
    if not path.exists():
        raise FileNotFoundError(f"Emotion map not found at {path}")

    with open(path) as f:
        raw = json.load(f)

    return {
        name: EmotionCoord(name=name, V=vals["V"], E=vals["E"], T=vals["T"])
        for name, vals in raw.items()
    }


# Module-level singletons — loaded once, importable everywhere
PARAMS = load_params()
EMOTION_MAP = load_emotion_map()
