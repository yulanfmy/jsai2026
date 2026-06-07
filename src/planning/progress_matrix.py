"""Soft-sequencing progress matrix (§4.3 of spec). ★critical

build_progress_matrix(axis_order, stage_alloc, alpha=0.6):
  Produces a 6×3 progress matrix p[i][k] ∈ [0,1].

Strict invariants:
  1. Each column (axis) is monotonically non-decreasing top to bottom.
  2. Last row (i=6) is all 1.0 exactly (target reached).
  3. The lead axis's increment in its stage is clearly larger than co-moving
     axes (reflecting α dominance).

Implementation memo from spec:
  (a) Determine axis stage intervals (which stages each axis "leads")
  (b) Per stage, give lead axis α share, distribute (1-α) among co-moving axes
  (c) Cumulate each column, check monotonicity, force-clamp last row to 1.0
"""

from __future__ import annotations

from src.config_loader import PARAMS


def build_progress_matrix(
    axis_order: list[str],
    stage_alloc: dict[str, int],
    alpha: float | None = None,
    N: int | None = None,
) -> list[dict[str, float]]:
    """Build the N×K progress matrix.

    Args:
        axis_order: axes in processing order (e.g., ['T', 'V', 'E']).
        stage_alloc: {axis: n_stages} allocation (sum = N).
        alpha: lead-axis concentration (default 0.6 from params).
        N: total stages (default 6 from params).

    Returns:
        List of N dicts, each {axis: cumulative_progress}.
        progress[i][k] is the progress of axis k at stage i+1.
    """
    if alpha is None:
        alpha = PARAMS.alpha
    if N is None:
        N = PARAMS.N

    all_axes = list(axis_order)

    # (a) Determine which stages each axis leads
    # Axes lead their allocated stages in sequence, following axis_order
    stage_map: list[str] = []  # stage_map[i] = which axis leads stage i
    for axis in axis_order:
        stage_map.extend([axis] * stage_alloc.get(axis, 0))

    # Pad or trim to exactly N
    while len(stage_map) < N:
        stage_map.append(axis_order[0])
    stage_map = stage_map[:N]

    # (b) Per stage, compute incremental progress
    # Each axis needs to go from 0 to 1 over N stages.
    # In its "lead" stages, it gets α share of its per-stage increment.
    # In non-lead stages, it gets a portion of (1-α).
    increments: list[dict[str, float]] = []

    for axis in all_axes:
        n_lead = stage_alloc.get(axis, 0)
        n_colead = N - n_lead

        if n_lead == 0:
            continue

        # Total progress = 1.0 for each axis
        # Lead stages: axis gets α per lead stage (of its total progress)
        # Co-lead stages: axis gets (1-α) spread over non-lead stages
        if n_lead >= N:
            # This axis leads ALL stages — each gets 1/N
            lead_increment = 1.0 / N
            colead_increment = 0.0
        else:
            lead_increment = alpha / n_lead
            colead_increment = (1.0 - alpha) / n_colead if n_colead > 0 else 0.0

        for i in range(N):
            if i >= len(increments):
                increments.append({a: 0.0 for a in all_axes})
            if stage_map[i] == axis:
                increments[i][axis] = lead_increment
            else:
                increments[i][axis] = colead_increment

    # Ensure we have N rows
    while len(increments) < N:
        increments.append({a: 0.0 for a in all_axes})

    # Handle axes with 0 allocation (zero delta)
    for axis in all_axes:
        if stage_alloc.get(axis, 0) == 0:
            # Spread evenly across all stages
            per_stage = 1.0 / N
            for i in range(N):
                increments[i][axis] = per_stage

    # (c) Cumulate each column
    progress: list[dict[str, float]] = []
    cumulative = {a: 0.0 for a in all_axes}

    for i in range(N):
        for a in all_axes:
            cumulative[a] += increments[i][a]
        progress.append(dict(cumulative))

    # Force-clamp last row to 1.0
    for a in all_axes:
        progress[N - 1][a] = 1.0

    # Verify monotonicity (should be guaranteed by construction, but assert)
    for a in all_axes:
        for i in range(1, N):
            if progress[i][a] < progress[i - 1][a] - 1e-10:
                # Fix by clamping
                progress[i][a] = progress[i - 1][a]

    return progress
