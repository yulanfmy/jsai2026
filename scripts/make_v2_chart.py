"""Generate the v2 LLM feature-estimation chart for the presentation slide.

Matches the style of the v1 slide 03 chart (horizontal bars, teal/orange).
Shows raw LLM vs Scheme 1+6 corrected for V and E.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ── Data from 5-fold CV leakage-safe evaluation (REPORT.md) ──
# Out-of-fold Pearson r on 4987 Zenodo tracks
data = [
    {"label": "energy (E)", "raw": 0.6267, "corrected": 0.7845},
    {"label": "happiness (V)", "raw": 0.3641, "corrected": 0.5499},
]

bar_height = 0.30
gap = 0.06  # gap between raw and corrected within a group
group_gap = 0.55  # gap between groups

fig, ax = plt.subplots(figsize=(9.5, 4.0))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

y_positions = []
y_labels = []
y_label_pos = []

for gi, d in enumerate(data):
    y_base = gi * (2 * bar_height + gap + group_gap)

    # Corrected bar (top = smaller y, shown higher after invert)
    y_cor = y_base
    y_raw = y_base + bar_height + gap

    y_positions.append((y_cor, y_raw))
    y_label_pos.append((y_cor + y_raw) / 2)  # center label between the two bars
    y_labels.append(d["label"])

    # Corrected bar
    ax.barh(y_cor, d["corrected"], height=bar_height,
            color="#3cc0ad", edgecolor="none")
    pct = (d["corrected"] - d["raw"]) / d["raw"] * 100
    ax.text(d["corrected"] + 0.015, y_cor,
            f".{round(d['corrected'] * 100):02d}",
            va="center", ha="left", fontsize=16, fontweight="bold", color="#333333")
    # improvement label
    ax.text(d["corrected"] + 0.09, y_cor,
            f"(+{pct:.0f}%)",
            va="center", ha="left", fontsize=12, fontweight="bold", color="#d35400")

    # Raw bar
    ax.barh(y_raw, d["raw"], height=bar_height,
            color="#f0a04b", edgecolor="none")
    ax.text(d["raw"] + 0.015, y_raw,
            f".{round(d['raw'] * 100):02d}",
            va="center", ha="left", fontsize=16, fontweight="bold", color="#333333")

ax.set_yticks(y_label_pos)
ax.set_yticklabels(y_labels, fontsize=15, fontweight="bold", color="#333333")
ax.set_xlim(0, 1.08)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis="x", labelsize=11, colors="#666666")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#cccccc")
ax.tick_params(axis="y", length=0)
ax.invert_yaxis()

# Legend
legend_elements = [
    Patch(facecolor="#3cc0ad", label="Scheme 1+6 corrected (5-fold CV, out-of-fold)"),
    Patch(facecolor="#f0a04b", label="Raw LLM (Gemini 2.5 Flash)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
          frameon=True, fancybox=True, shadow=False, edgecolor="#dddddd")

# Subtitle
fig.text(0.5, 0.99,
         "Pearson r vs Zenodo ground truth  ·  4,987 tracks  ·  5-fold CV (leakage-safe)",
         ha="center", va="top", fontsize=10, color="#888888", style="italic")

# Note about tension
fig.text(0.12, 0.02,
         "tension (T): ground truth unavailable — LLM direct estimate used",
         ha="left", va="bottom", fontsize=9, color="#aaaaaa", style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
out = "/home/ubuntu/repos/jsai2026/eval/llm_feature_eval/outputs/v2_feature_chart.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {out}")
plt.close()
