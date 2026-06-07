"""Analyze A/B/C blind evaluation results: Dynamic vs Linear vs Auto.

Paired comparison within each test_id (same evaluator, same transition).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.patches import Patch

# ── Load data ──
df = pd.read_csv("/home/ubuntu/eval_data.csv")
df.columns = df.columns.str.strip()
df["Method"] = df["Method"].str.strip()
df["Name"] = df["Name"].str.strip()
df["From"] = df["From"].str.strip()
df["To"] = df["To"].str.strip()
df["Score"] = pd.to_numeric(df["Score"])

# Anonymize evaluator names
name_map = {n: f"Evaluator {chr(65+i)}" for i, n in enumerate(sorted(df["Name"].unique()))}
df["Name"] = df["Name"].map(name_map)

OUT = "/home/ubuntu/repos/jsai2026/eval/llm_feature_eval/outputs"
method_order = ["Dynamic", "Linear", "Auto"]
colors = {"Dynamic": "#3cc0ad", "Linear": "#f0a04b", "Auto": "#e07070"}

# ── Build paired comparison table ──
test_ids = sorted(df["test_id"].unique())
n_tests = len(test_ids)

rows = []
for tid in test_ids:
    td = df[df["test_id"] == tid]
    name = td["Name"].iloc[0]
    fr = td["From"].iloc[0]
    to = td["To"].iloc[0]
    n = int(td["N"].iloc[0])
    s = {m: td[td["Method"] == m]["Score"].values[0] for m in method_order}
    # Winner for this test
    best_score = max(s.values())
    winners = [m for m, v in s.items() if v == best_score]
    rows.append({
        "test_id": tid, "Name": name, "From": fr, "To": to, "N": n,
        "Dynamic": s["Dynamic"], "Linear": s["Linear"], "Auto": s["Auto"],
        "Winner": "/".join(winners),
    })
paired_df = pd.DataFrame(rows)

print("=" * 70)
print("PAIRED COMPARISON (within each test_id)")
print("=" * 70)
print()
print(paired_df.to_string(index=False))
print()

# ── Win counts ──
print("Winner counts:")
print("-" * 40)
for m in method_order:
    wins = sum(1 for _, r in paired_df.iterrows() if r["Winner"] == m)
    shared = sum(1 for _, r in paired_df.iterrows() if m in r["Winner"] and r["Winner"] != m)
    print(f"  {m:8s}: {wins} solo wins, {shared} shared wins")

# ── Rank within each test ──
print("\nRank distribution (1=best, 3=worst):")
print("-" * 40)
rank_counts = {m: {1: 0, 2: 0, 3: 0} for m in method_order}
for _, r in paired_df.iterrows():
    scores = [(r[m], m) for m in method_order]
    scores.sort(key=lambda x: -x[0])
    prev_score = None
    prev_rank = 0
    for i, (sc, meth) in enumerate(scores):
        if sc != prev_score:
            rank = i + 1
        rank_counts[meth][rank] += 1
        prev_score = sc
        prev_rank = rank

for m in method_order:
    r = rank_counts[m]
    print(f"  {m:8s}: 1st={r[1]}, 2nd={r[2]}, 3rd={r[3]}")

# ── Overall averages ──
print("\nOverall averages:")
print("-" * 40)
for m in method_order:
    vals = paired_df[m].values
    print(f"  {m:8s}: mean={vals.mean():.3f}, std={vals.std():.3f}, median={np.median(vals):.1f}")

# ── Statistical tests (paired) ──
print("\nStatistical Tests (Wilcoxon signed-rank, paired by test_id):")
print("-" * 60)
for i in range(len(method_order)):
    for j in range(i + 1, len(method_order)):
        m1, m2 = method_order[i], method_order[j]
        s1 = paired_df[m1].values
        s2 = paired_df[m2].values
        diffs = s1 - s2
        nonzero = np.sum(diffs != 0)
        if nonzero >= 2:
            stat, p = stats.wilcoxon(s1, s2, alternative="two-sided")
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
            print(f"  {m1} vs {m2}: W={stat:.1f}, p={p:.4f} {sig}  "
                  f"(mean diff={diffs.mean():+.3f}, median diff={np.median(diffs):+.1f})")
        else:
            print(f"  {m1} vs {m2}: too few non-tied pairs ({nonzero})")

# ═══════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════

# ── Chart 1: Per-test grouped bars (primary chart) ──
fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bar_w = 0.25
x_pos = np.arange(n_tests)

for i, m in enumerate(method_order):
    offset = (i - 1) * (bar_w + 0.02)
    vals = paired_df[m].values
    bars = ax.bar(x_pos + offset, vals, width=bar_w, color=colors[m],
                  label=m, edgecolor="none", alpha=0.85)
    for j, v in enumerate(vals):
        ax.text(x_pos[j] + offset, v + 0.05, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#555555")

# Mark winner(s) with a star
for j, (_, r) in enumerate(paired_df.iterrows()):
    best = max(r["Dynamic"], r["Linear"], r["Auto"])
    for i, m in enumerate(method_order):
        if r[m] == best:
            offset = (i - 1) * (bar_w + 0.02)
            ax.text(x_pos[j] + offset, r[m] + 0.25, "★",
                    ha="center", va="bottom", fontsize=10, color="#d4a017")

# X labels
xlabels = [f"T{r['test_id']}\n{r['From']}→{r['To']}\n(N={r['N']})" for _, r in paired_df.iterrows()]
ax.set_xticks(x_pos)
ax.set_xticklabels(xlabels, fontsize=8, ha="center")
ax.set_ylabel("Score (1–5)", fontsize=12)
ax.set_ylim(0, 5.9)
ax.set_yticks([1, 2, 3, 4, 5])
ax.legend(fontsize=10, loc="upper right", frameon=True, edgecolor="#dddddd")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")
ax.yaxis.grid(True, alpha=0.3, linestyle="--")
ax.set_title("Paired Comparison: Same Evaluator × Same Transition  (★ = winner)",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(f"{OUT}/eval_per_test.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUT}/eval_per_test.png")
plt.close()

# ── Chart 2: Overall bar chart with individual dots ──
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

means = [paired_df[m].mean() for m in method_order]
stds_vals = [paired_df[m].std() for m in method_order]
x_pos = np.arange(len(method_order))

bars = ax.bar(x_pos, means, width=0.55, color=[colors[m] for m in method_order],
              edgecolor="none", alpha=0.85, zorder=2)
ax.errorbar(x_pos, means, yerr=stds_vals, fmt="none", ecolor="#555555",
            capsize=6, capthick=1.5, linewidth=1.5, zorder=3)

# Individual dots
np.random.seed(42)
for i, m in enumerate(method_order):
    scores = paired_df[m].values
    jitter = np.random.normal(0, 0.06, size=len(scores))
    ax.scatter(np.full_like(scores, i, dtype=float) + jitter, scores,
               color="white", edgecolors="#333333", linewidth=1, s=40,
               zorder=4, alpha=0.8)

for i, (m, mean_val) in enumerate(zip(method_order, means)):
    ax.text(i, mean_val + stds_vals[i] + 0.15, f"{mean_val:.2f}",
            ha="center", va="bottom", fontsize=14, fontweight="bold", color="#333333")

ax.set_xticks(x_pos)
ax.set_xticklabels(["Dynamic\n(v2 Viterbi DP)", "Linear\n(baseline)", "Spotify\nAutoplay"],
                   fontsize=12, fontweight="bold")
ax.set_ylabel("Score (1–5)", fontsize=13)
ax.set_ylim(0, 5.8)
ax.set_yticks([1, 2, 3, 4, 5])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")
ax.yaxis.grid(True, alpha=0.3, linestyle="--")
fig.text(0.5, 0.97,
         f"Blind A/B/C Evaluation  ·  {n_tests} paired tests  ·  {len(df['Name'].unique())} evaluators",
         ha="center", va="top", fontsize=11, color="#888888", style="italic")
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/eval_method_comparison.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}/eval_method_comparison.png")
plt.close()

# ── Chart 3: Per-evaluator grouped bars ──
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

evaluators = sorted(df["Name"].unique())
n_eval = len(evaluators)
bar_w = 0.22
x_pos = np.arange(n_eval)

for i, m in enumerate(method_order):
    vals = [df[(df["Name"] == e) & (df["Method"] == m)]["Score"].mean() for e in evaluators]
    offset = (i - 1) * (bar_w + 0.03)
    ax.bar(x_pos + offset, vals, width=bar_w, color=colors[m],
           label=m, edgecolor="none", alpha=0.85)
    for j, v in enumerate(vals):
        ax.text(x_pos[j] + offset, v + 0.08, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#555555")

ax.set_xticks(x_pos)
ax.set_xticklabels(evaluators, fontsize=13, fontweight="bold")
ax.set_ylabel("Average Score", fontsize=12)
ax.set_ylim(0, 5.5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.legend(fontsize=11, loc="upper right", frameon=True, edgecolor="#dddddd")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")
ax.yaxis.grid(True, alpha=0.3, linestyle="--")
ax.set_title("Per-Evaluator Average Scores", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(f"{OUT}/eval_per_evaluator.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}/eval_per_evaluator.png")
plt.close()

# ── Chart 4: Paired difference chart (Dynamic - others) ──
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
fig.patch.set_facecolor("white")

for ax_i, (other, ax) in enumerate(zip(["Linear", "Auto"], axes)):
    ax.set_facecolor("white")
    diffs = paired_df["Dynamic"].values - paired_df[other].values
    x = np.arange(n_tests)

    bar_colors = ["#3cc0ad" if d > 0 else "#e07070" if d < 0 else "#cccccc" for d in diffs]
    ax.bar(x, diffs, width=0.6, color=bar_colors, edgecolor="none", alpha=0.85)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="-")

    # Mean diff line
    mean_diff = diffs.mean()
    ax.axhline(mean_diff, color="#d35400", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.text(n_tests - 0.5, mean_diff + 0.1, f"mean={mean_diff:+.2f}",
            ha="right", va="bottom", fontsize=9, color="#d35400", fontweight="bold")

    xlabels = [f"T{r['test_id']}" for _, r in paired_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_title(f"Dynamic − {other}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score Difference" if ax_i == 0 else "", fontsize=11)
    ax.set_ylim(-2.5, 3.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")

legend_elements = [
    Patch(facecolor="#3cc0ad", label="Dynamic wins"),
    Patch(facecolor="#e07070", label="Dynamic loses"),
    Patch(facecolor="#cccccc", label="Tie"),
]
axes[1].legend(handles=legend_elements, fontsize=9, loc="lower right",
               frameon=True, edgecolor="#dddddd")

fig.suptitle("Paired Differences (Dynamic vs each baseline)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT}/eval_paired_diff.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}/eval_paired_diff.png")
plt.close()

print("\nDone.")
