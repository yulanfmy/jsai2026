"""Stage allocation — |Δ|-proportional distribution over N=6 stages (§4.2).

allocate_stages(delta, axis_order, N=6):
  Allocates stages proportionally to |Δ| per axis.
  Rounding remainder added to the largest-|Δ| axis.
  Sum guaranteed = N.  No axis gets 0 if it has nonzero |Δ|.
"""

from __future__ import annotations

from src.config_loader import PARAMS


def allocate_stages(
    delta: dict[str, float],
    axis_order: list[str],
    N: int | None = None,
) -> dict[str, int]:
    """Allocate N stages proportionally to |Δ| per axis.

    Args:
        delta: {V, E, T} delta values.
        axis_order: axes in processing order (used to determine the lead axis).
        N: total stages (default from params.yaml).

    Returns:
        {V: n_v, E: n_e, T: n_t} where sum = N.
    """
    if N is None:
        N = PARAMS.N

    abs_deltas = {a: abs(delta[a]) for a in axis_order}
    total_delta = sum(abs_deltas.values())

    if total_delta == 0:
        # No change needed — distribute equally
        base = N // len(axis_order)
        alloc = {a: base for a in axis_order}
        remainder = N - sum(alloc.values())
        for a in axis_order[:remainder]:
            alloc[a] += 1
        return alloc

    # Proportional allocation with rounding
    raw = {a: (abs_deltas[a] / total_delta) * N for a in axis_order}
    alloc = {a: max(1, round(raw[a])) if abs_deltas[a] > 0 else 0 for a in axis_order}

    # Ensure axes with nonzero Δ get at least 1 stage
    for a in axis_order:
        if abs_deltas[a] > 0 and alloc[a] == 0:
            alloc[a] = 1

    # Adjust to sum to N
    current = sum(alloc.values())

    if current < N:
        # Add remainder to the largest-|Δ| axis
        lead_axis = axis_order[0]
        alloc[lead_axis] += N - current
    elif current > N:
        # Remove excess from smallest-|Δ| axes first
        for a in reversed(axis_order):
            while current > N and alloc[a] > 1:
                alloc[a] -= 1
                current -= 1

    return alloc
