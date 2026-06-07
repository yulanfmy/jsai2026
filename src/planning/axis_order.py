"""Δ computation and axis ordering with start-state protection (§4.1).

Implements:
  compute_delta(source, target) → {V, E, T} delta values
  compute_axis_order(source, target, source_label) → axes sorted by |Δ|,
    with start-state protection override: if source_label ∈ {Angry, Anxious, Fear},
    force T axis to front regardless of |ΔT| magnitude.
"""

from __future__ import annotations

from src.config_loader import PARAMS
from src.emotions import Emotion

START_PROTECTION = set(PARAMS.start_protection)


def compute_delta(source: Emotion, target: Emotion) -> dict[str, float]:
    """Compute per-axis deltas from source to target."""
    return {
        "V": target.V - source.V,
        "E": target.E - source.E,
        "T": target.T - source.T,
    }


def compute_axis_order(
    source: Emotion,
    target: Emotion,
    source_label: str,
) -> tuple[list[str], dict[str, float]]:
    """Compute the axis processing order.

    Returns (axis_order, delta).

    axis_order: axes sorted by |Δ| descending.
    If source_label ∈ START_PROTECTION, T is forced to front.
    """
    delta = compute_delta(source, target)

    # Sort by |Δ| descending
    axes = sorted(delta.keys(), key=lambda a: abs(delta[a]), reverse=True)

    # Start-state protection: force T to front for high-T emotions
    if source_label in START_PROTECTION and "T" in axes:
        axes.remove("T")
        axes.insert(0, "T")

    return axes, delta
