"""Viterbi DP global selection (§7 of spec). ★most critical

Forward recursion + back-pointers + backtrack.

KEY INVARIANTS (from spec):
  - cost[i][b] stored separately per candidate b at each stage
  - min taken over predecessor a, not fixing b per stage
  - Bellman: whichever b chosen, best predecessor is consistent at backtrack
  - Different from greedy, which fixes choices step by step
  - λ multiplies transition term AS A WHOLE (not inside transition)
  - Stage 1 has NO transition term (no predecessor exists)

transition(a, b) = w1·(1 - key_compat) + w2·(1 - tempo_compat) + w3·arc_join
"""

from __future__ import annotations

import math

from src.config_loader import PARAMS
from src.planning.targets import StageTarget
from src.selection.candidates import target_loss
from src.selection.continuity import key_compat, tempo_compat, arc_join


def _transition_cost(a: dict, b: dict) -> float:
    """Compute transition cost between two adjacent tracks.

    transition = w1·(1 - key_compat) + w2·(1 - tempo_compat) + w3·arc_join
    """
    w1, w2, w3 = PARAMS.w1, PARAMS.w2, PARAMS.w3

    kc = key_compat(
        int(a.get("key", 0)), int(a.get("mode", 0)),
        int(b.get("key", 0)), int(b.get("mode", 0)),
    )
    tc = tempo_compat(
        float(a.get("tempo", 120)),
        float(b.get("tempo", 120)),
    )
    aj = arc_join(
        float(a.get("arc_end_E", a.get("E", 0))),
        float(a.get("arc_end_T", a.get("T", 0))),
        float(b.get("arc_start_E", b.get("E", 0))),
        float(b.get("arc_start_T", b.get("T", 0))),
    )

    return w1 * (1.0 - kc) + w2 * (1.0 - tc) + w3 * aj


def viterbi_select(
    candidates: list[list[dict]],
    stages: list[StageTarget],
    lam: float | None = None,
    gamma: float | None = None,
) -> list[dict]:
    """Viterbi DP — forward recursion + backtrack.

    Args:
        candidates: K candidates per stage (from top_k_filter).
        stages: N StageTarget objects.
        lam: transition weight λ (default from params).
        gamma: arc-direction weight γ (default from params).

    Returns:
        List of N selected tracks (one per stage), globally optimized.
    """
    if lam is None:
        lam = PARAMS.lam
    if gamma is None:
        gamma = PARAMS.gamma

    N = len(stages)
    if N == 0:
        return []

    # cost[i][j] = minimum total cost to reach candidate j at stage i
    # back[i][j] = index of best predecessor at stage i-1
    cost: list[list[float]] = []
    back: list[list[int]] = []

    # Stage 1: no transition term
    stage1_costs: list[float] = []
    for j, cand in enumerate(candidates[0]):
        c = target_loss(cand, stages[0], gamma)
        stage1_costs.append(c)
    cost.append(stage1_costs)
    back.append([-1] * len(candidates[0]))  # no predecessor

    # Stages 2..N: forward recursion
    for i in range(1, N):
        stage_costs: list[float] = []
        stage_back: list[int] = []

        for j, cand_j in enumerate(candidates[i]):
            tl_j = target_loss(cand_j, stages[i], gamma)

            # Find best predecessor a at stage i-1
            best_total = math.inf
            best_a = 0

            for a, cand_a in enumerate(candidates[i - 1]):
                # Bellman: cost[i][j] = tl_j + min_a(cost[i-1][a] + λ·transition(a,j))
                trans = _transition_cost(cand_a, cand_j)
                total = cost[i - 1][a] + lam * trans
                if total < best_total:
                    best_total = total
                    best_a = a

            stage_costs.append(tl_j + best_total)
            stage_back.append(best_a)

        cost.append(stage_costs)
        back.append(stage_back)

    # Backtrack: find best candidate at last stage
    last_costs = cost[N - 1]
    best_last = min(range(len(last_costs)), key=lambda j: last_costs[j])

    # Trace back
    path: list[int] = [0] * N
    path[N - 1] = best_last
    for i in range(N - 2, -1, -1):
        path[i] = back[i + 1][path[i + 1]]

    # Extract selected tracks
    selected = [candidates[i][path[i]] for i in range(N)]

    return selected


def greedy_select(
    candidates: list[list[dict]],
    stages: list[StageTarget],
    gamma: float | None = None,
) -> list[dict]:
    """Greedy baseline (for comparison with Viterbi).

    At each stage, pick the candidate with the lowest target_loss
    independently, without considering transition costs.
    """
    if gamma is None:
        gamma = PARAMS.gamma

    selected: list[dict] = []
    for i, stage in enumerate(stages):
        best = min(candidates[i], key=lambda t: target_loss(t, stage, gamma))
        selected.append(best)
    return selected
