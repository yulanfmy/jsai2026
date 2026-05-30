"""Stage targets and expected direction computation (§4.4).

compute_targets(source, target, delta, progress_matrix, axis_order, stage_alloc):
  Outputs 6 StageTarget objects with (V, E, T) coordinates and
  expected_dir (E, T) plane vectors for arc matching.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.emotions import Emotion


@dataclass
class StageTarget:
    stage: int          # 1-based stage number
    V: float
    E: float
    T: float
    expected_dir_E: float  # expected arc direction in E
    expected_dir_T: float  # expected arc direction in T
    lead_axis: str         # which axis leads this stage
    explanation: str = ""


def compute_targets(
    source: Emotion,
    target: Emotion,
    delta: dict[str, float],
    progress_matrix: list[dict[str, float]],
    axis_order: list[str],
    stage_alloc: dict[str, int],
) -> list[StageTarget]:
    """Compute target coordinates and expected directions for each stage.

    target_i = source + delta * progress[i]
    expected_dir_i = target_i - target_{i-1}  (in E,T plane)
    For stage 1, the "previous point" is the source.
    """
    N = len(progress_matrix)

    # Determine which axis leads each stage
    stage_leaders: list[str] = []
    for axis in axis_order:
        stage_leaders.extend([axis] * stage_alloc.get(axis, 0))
    while len(stage_leaders) < N:
        stage_leaders.append(axis_order[0])
    stage_leaders = stage_leaders[:N]

    targets: list[StageTarget] = []
    prev_V = source.V
    prev_E = source.E
    prev_T = source.T

    for i in range(N):
        p = progress_matrix[i]

        # target_i = source + delta * progress[i]
        t_V = source.V + delta["V"] * p["V"]
        t_E = source.E + delta["E"] * p["E"]
        t_T = source.T + delta["T"] * p["T"]

        # expected_dir = target_i - previous_point (in E,T plane)
        dir_E = t_E - prev_E
        dir_T = t_T - prev_T

        lead = stage_leaders[i]
        explanation = f"Stage {i+1}: lead axis {lead}, target ({t_V:.2f}, {t_E:.2f}, {t_T:.2f})"

        targets.append(StageTarget(
            stage=i + 1,
            V=t_V, E=t_E, T=t_T,
            expected_dir_E=dir_E,
            expected_dir_T=dir_T,
            lead_axis=lead,
            explanation=explanation,
        ))

        prev_V, prev_E, prev_T = t_V, t_E, t_T

    return targets
