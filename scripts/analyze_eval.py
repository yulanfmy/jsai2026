"""Analyze A/B/C blind evaluation results: Dynamic vs Linear vs Auto."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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

print("=" * 60)
print("BLIND A/B/C EVALUATION RESULTS")
print("=" * 60)
print(f"\nTotal evaluations: {len(df)} ({len(df)//3} test sessions × 3 methods)")
print(f"Evaluators: {', '.join(df['Name'].unique())}")
print(f"Emotion transitions tested: {df.groupby(['From','To']).ngroups}")
print()

# ── Overall averages ──
method_stats = df.groupby("Method")["Score"].agg(["mean", "std", "count", "median"])
method_stats = method_stats.sort_values("mean", ascending=False)
print("Overall Method Averages (sorted by mean):")
print("-" * 50)
for method, row in method_stats.iterrows():
    print(f"  {method:8s}  mean={row['mean']:.3f}  std={row['std']:.3f}  "
          f"median={row['median']:.1f}  n={int(row['count'])}")

# ── Per evaluator ──
print("\nPer-Evaluator Breakdown:")
print("-" * 50)
pivot_eval = df.pivot_table(values="Score", index="Name", columns="Method", aggfunc="mean")
print(pivot_eval.round(3).to_string())

# ── Per test (wide format) ──
print("\nPer-Test Scores:")
print("-" * 50)
pivot_test = df.pivot_table(values="Score", index=["test_id", "Name", "From", "To", "N"],
                            columns="Method", aggfunc="first")
print(pivot_test.to_string())

# ── Win/tie/lose counts ──
print("\nHead-to-Head Comparison (Dynamic vs others):")
print("-" * 50)
for other in ["Linear", "Auto"]:
    wins = ties = losses = 0
    for tid in df["test_id"].unique():
        d_score = df[(df["test_id"] == tid) & (df["Method"] == "Dynamic")]["Score"].values[0]
        o_score = df[(df["test_id"] == tid) & (df["Method"] == other)]["Score"].values[0]
        if d_score > o_score:
            wins += 1
        elif d_score == o_score:
            ties += 1
        else:
            losses += 1
    print(f"  Dynamic vs {other:6s}: {wins}W / {ties}T / {losses}L")

# ── Statistical tests ──
print("\nStatistical Tests (Wilcoxon signed-rank, paired):")
print("-" * 50)
methods = ["Dynamic", "Linear", "Auto"]
for i in range(len(methods)):
    for j in range(i + 1, len(methods)):
        m1, m2 = methods[i], methods[j]
        scores1, scores2 = [], []
        for tid in sorted(df["test_id"].unique()):
            s1 = df[(df["test_id"] == tid) & (df["Method"] == m1)]["Score"].values[0]
            s2 = df[(df["test_id"] == tid) & (df["Method"] == m2)]["Score"].values[0]
            scores1.append(s1)
            scores2.append(s2)
        diffs = [a - b for a, b in zip(scores1, scores2)]
        nonzero_diffs = [d for d in diffs if d != 0]
        if len(nonzero_diffs) >= 2:
            stat, p = stats.wilcoxon(scores1, scores2, alternative="two-sided")
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
            print(f"  {m1} vs {m2}: W={stat:.1f}, p={p:.4f} {sig}  (mean diff={np.mean(diffs):+.3f})")
        else:
            print(f"  {m1} vs {m2}: too few non-tied pairs ({len(nonzero_diffs)})")

# ═══════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════
OUT = "/home/ubuntu/repos/jsai2026/eval/llm_feature_eval/outputs"

colors = {"Dynamic": "#3cc0ad", "Linear": "#f0a04b", "Auto": "#e07070"}

# ── Chart 1: Overall bar chart with individual dots ──
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

method_order = ["Dynamic", "Linear", "Auto"]
means = [df[df["Method"] == m]["Score"].mean() for m in method_order]
stds = [df[df["Method"] == m]["Score"].std() for m in method_order]
x_pos = np.arange(len(method_order))

bars = ax.bar(x_pos, means, width=0.55, color=[colors[m] for m in method_order],
              edgecolor="none", alpha=0.85, zorder=2)
ax.errorbar(x_pos, means, yerr=stds, fmt="none", ecolor="#555555",
            capsize=6, capthick=1.5, linewidth=1.5, zorder=3)

# Scatter individual scores with jitter
np.random.seed(42)
for i, m in enumerate(method_order):
    scores = df[df["Method"] == m]["Score"].values
    jitter = np.random.normal(0, 0.06, size=len(scores))
    ax.scatter(np.full_like(scores, i, dtype=float) + jitter, scores,
               color="white", edgecolors="#333333", linewidth=1, s=40,
               zorder=4, alpha=0.8)

# Labels
for i, (m, mean_val) in enumerate(zip(method_order, means)):
    ax.text(i, mean_val + stds[i] + 0.15, f"{mean_val:.2f}",
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
         f"Blind A/B/C Evaluation  ·  {len(df)//3} sessions  ·  {len(df['Name'].unique())} evaluators",
         ha="center", va="top", fontsize=11, color="#888888", style="italic")
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/eval_method_comparison.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUT}/eval_method_comparison.png")
plt.close()

# ── Chart 2: Per-evaluator grouped bar chart ──
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
    bars = ax.bar(x_pos + offset, vals, width=bar_w, color=colors[m],
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

# ── Chart 3: Per-test heatmap-style comparison ──
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

test_labels = []
scores_matrix = []
for tid in sorted(df["test_id"].unique()):
    row_data = df[df["test_id"] == tid]
    name = row_data["Name"].iloc[0]
    fr = row_data["From"].iloc[0]
    to = row_data["To"].iloc[0]
    n = int(row_data["N"].iloc[0])
    test_labels.append(f"T{tid}\n{fr}→{to} (N={n})")
    row_scores = []
    for m in method_order:
        s = row_data[row_data["Method"] == m]["Score"].values[0]
        row_scores.append(s)
    scores_matrix.append(row_scores)

scores_arr = np.array(scores_matrix)
n_tests = len(test_labels)
bar_w = 0.25
x_pos = np.arange(n_tests)

for i, m in enumerate(method_order):
    offset = (i - 1) * (bar_w + 0.02)
    ax.bar(x_pos + offset, scores_arr[:, i], width=bar_w, color=colors[m],
           label=m, edgecolor="none", alpha=0.85)

ax.set_xticks(x_pos)
ax.set_xticklabels(test_labels, fontsize=8, ha="center")
ax.set_ylabel("Score (1–5)", fontsize=12)
ax.set_ylim(0, 5.8)
ax.set_yticks([1, 2, 3, 4, 5])
ax.legend(fontsize=10, loc="upper right", frameon=True, edgecolor="#dddddd")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")
ax.yaxis.grid(True, alpha=0.3, linestyle="--")
ax.set_title("Per-Test Score Comparison", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(f"{OUT}/eval_per_test.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}/eval_per_test.png")
plt.close()

print("\nDone.")
