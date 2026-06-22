"""
Evaluation Pipeline v2 — Analysis & Figure Generation

Reads frame_results.csv -> produces all statistics and thesis-ready figures.
One figure per research question. No redundant table+figure pairs.

Usage:
    python evaluation_v2/analyze.py
    python evaluation_v2/analyze.py --results evaluation_v2/results/rehab24/frame_results.csv
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_v2.config import COMPARABLE_JOINT_NAMES, JOINT_GROUPS

# =============================================================================
# Style
# =============================================================================

MODEL_COLORS = {
    "MediaPipe": "#2196F3",
    "MoveNet": "#4CAF50",
    "YOLOv8": "#FF5722",
}
MODEL_ORDER = ["MediaPipe", "MoveNet", "YOLOv8"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.figsize": (7, 4.5),
    "figure.constrained_layout.use": True,
})


# =============================================================================
# Data Loading & Filtering
# =============================================================================

def load_and_filter(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load results and create standard subsets.

    Returns:
        (df_all, df_clean, df_valid)
        - df_all: everything
        - df_clean: no detection failures
        - df_valid: no detection failures, no outliers, no mocap errors
    """
    df = pd.read_csv(csv_path)
    df_clean = df[~df["is_detection_failure"]].copy()
    df_valid = df_clean[
        ~df_clean["is_outlier"] & (df_clean["mocap_erroneous"] == 0)
    ].copy()

    print(f"Loaded {len(df):,} rows")
    print(f"  Clean (no detection failure): {len(df_clean):,}")
    print(f"  Valid (no outlier, no mocap error): {len(df_valid):,}")
    return df, df_clean, df_valid


# =============================================================================
# 1. Overall Accuracy
# =============================================================================

def analyze_accuracy(df: pd.DataFrame, fig_dir: Path):
    """Overall NMPJPE comparison across models."""
    print("\n--- Overall Accuracy ---")

    summary = []
    for model in MODEL_ORDER:
        m = df[df["model"] == model]["nmpjpe"]
        row = {
            "model": model,
            "mean": m.mean(),
            "std": m.std(),
            "median": m.median(),
            "n_frames": len(m),
        }
        summary.append(row)
        print(f"  {model:12s}: {row['mean']:.1f}% ± {row['std']:.1f}% (median {row['median']:.1f}%, n={row['n_frames']:,})")

    # Pairwise Wilcoxon signed-rank tests (matched by frame)
    print("\n  Pairwise Wilcoxon signed-rank tests:")
    models = MODEL_ORDER
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = df[df["model"] == models[i]].set_index(["video_id", "exercise", "camera", "frame_idx"])["nmpjpe"]
            b = df[df["model"] == models[j]].set_index(["video_id", "exercise", "camera", "frame_idx"])["nmpjpe"]
            common = a.index.intersection(b.index)
            if len(common) > 0:
                stat, p = stats.wilcoxon(a.loc[common], b.loc[common])
                d = (a.loc[common].mean() - b.loc[common].mean()) / a.loc[common].std()
                print(f"    {models[i]} vs {models[j]}: p={p:.2e}, Cohen's d={d:.3f}, n={len(common):,}")

    # Figure: bar chart with error bars
    fig, ax = plt.subplots()
    x = np.arange(len(models))
    means = [s["mean"] for s in summary]
    stds = [s["std"] for s in summary]
    medians = [s["median"] for s in summary]
    colors = [MODEL_COLORS[m] for m in models]

    bars = ax.bar(x - 0.15, means, 0.3, yerr=stds, capsize=4, color=colors, alpha=0.8, label="Mean ± SD")
    ax.bar(x + 0.15, medians, 0.3, color=colors, alpha=0.4, label="Median")

    ax.set_ylabel("NMPJPE (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, max(means) + max(stds) + 5)
    ax.set_title("Overall Accuracy (lower is better)")

    for i, (mean, med) in enumerate(zip(means, medians)):
        ax.text(i - 0.15, mean + stds[i] + 0.5, f"{mean:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.15, med + 0.5, f"{med:.1f}", ha="center", fontsize=8)

    fig.savefig(fig_dir / "fig1_accuracy.pdf")
    fig.savefig(fig_dir / "fig1_accuracy.png")
    plt.close(fig)
    print(f"  Saved fig1_accuracy")

    return pd.DataFrame(summary)


# =============================================================================
# 2. Rotation Robustness
# =============================================================================

def analyze_rotation(df: pd.DataFrame, fig_dir: Path):
    """NMPJPE by rotation angle bin."""
    print("\n--- Rotation Robustness ---")

    bins = sorted(df["rotation_bin"].unique())
    fig, ax = plt.subplots()

    for model in MODEL_ORDER:
        m = df[df["model"] == model]
        means = [m[m["rotation_bin"] == b]["nmpjpe"].mean() for b in bins]
        counts = [len(m[m["rotation_bin"] == b]) for b in bins]
        label = f"{model} (n={len(m):,})"
        ax.plot(
            [b + 5 for b in bins], means,
            "o-", color=MODEL_COLORS[model], label=label, markersize=4,
        )
        # Print frontal vs lateral
        frontal = m[m["rotation_bin"] <= 20]["nmpjpe"].mean()
        lateral = m[m["rotation_bin"] >= 60]["nmpjpe"].mean()
        change = (lateral / frontal - 1) * 100
        print(f"  {model:12s}: frontal {frontal:.1f}% -> lateral {lateral:.1f}% ({change:+.0f}%)")

    ax.set_xlabel("Camera-Relative Rotation (°)")
    ax.set_ylabel("Mean NMPJPE (%)")
    ax.set_title("Accuracy by Body Rotation")
    ax.legend(fontsize=8)
    ax.set_xticks([b + 5 for b in bins])
    ax.set_xticklabels([f"{b}-{b+10}" for b in bins], fontsize=7, rotation=45)

    fig.savefig(fig_dir / "fig2_rotation.pdf")
    fig.savefig(fig_dir / "fig2_rotation.png")
    plt.close(fig)
    print(f"  Saved fig2_rotation")


# =============================================================================
# 3. Multi-Person Robustness (frame-level!)
# =============================================================================

def analyze_multi_person(df: pd.DataFrame, fig_dir: Path):
    """Compare NMPJPE on single-person vs multi-person frames."""
    print("\n--- Multi-Person Robustness (frame-level) ---")

    clean = df[~df["is_multi_person"]]
    multi = df[df["is_multi_person"]]

    print(f"  Single-person frames: {len(clean):,}")
    print(f"  Multi-person frames:  {len(multi):,}")

    if len(multi) == 0:
        print("  No multi-person frames found. Skipping.")
        return

    models = MODEL_ORDER
    clean_means = [clean[clean["model"] == m]["nmpjpe"].mean() for m in models]
    multi_means = [multi[multi["model"] == m]["nmpjpe"].mean() for m in models]

    fig, ax = plt.subplots()
    x = np.arange(len(models))
    w = 0.35

    ax.bar(x - w/2, clean_means, w, color=[MODEL_COLORS[m] for m in models], alpha=0.7, label="Single-person")
    ax.bar(x + w/2, multi_means, w, color=[MODEL_COLORS[m] for m in models], alpha=0.3, label="Multi-person", edgecolor=[MODEL_COLORS[m] for m in models], linewidth=1.5)

    for i, (c, mp) in enumerate(zip(clean_means, multi_means)):
        change = (mp / c - 1) * 100
        ax.text(i + w/2, mp + 1, f"+{change:.0f}%", ha="center", fontsize=8, fontweight="bold")

    ax.set_ylabel("Mean NMPJPE (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_title("Multi-Person Impact (frame-level analysis)")

    fig.savefig(fig_dir / "fig3_multiperson.pdf")
    fig.savefig(fig_dir / "fig3_multiperson.png")
    plt.close(fig)

    for model, c, mp in zip(models, clean_means, multi_means):
        print(f"  {model:12s}: clean {c:.1f}% -> multi {mp:.1f}% (+{(mp/c-1)*100:.0f}%)")
    print(f"  Saved fig3_multiperson")


# =============================================================================
# 4. Temporal Stability (Jitter)
# =============================================================================

def analyze_jitter(results_dir: Path, fig_dir: Path):
    """Temporal jitter analysis from jitter_results.csv."""
    print("\n--- Temporal Stability ---")

    jitter_path = results_dir / "jitter_results.csv"
    if not jitter_path.exists():
        print("  No jitter data found. Skipping.")
        return

    jdf = pd.read_csv(jitter_path)
    jitter_col = "keypoint_jitter" if "keypoint_jitter" in jdf.columns else "nmpjpe_jitter"
    y_label = (
        "Frame-to-Frame Keypoint Jitter (% torso)"
        if jitter_col == "keypoint_jitter"
        else "Frame-to-Frame Jitter (NMPJPE %)"
    )
    title = (
        "Temporal Stability (true keypoint displacement)"
        if jitter_col == "keypoint_jitter"
        else "Temporal Stability (lower is smoother)"
    )

    fig, ax = plt.subplots()
    data_to_plot = []
    labels = []
    colors = []

    for model in MODEL_ORDER:
        m = jdf[jdf["model"] == model][jitter_col]
        data_to_plot.append(m.values)
        labels.append(model)
        colors.append(MODEL_COLORS[model])
        print(f"  {model:12s}: mean {m.mean():.2f}%, median {m.median():.2f}%, P90 {m.quantile(0.9):.2f}%")

    bp = ax.boxplot(
        data_to_plot, labels=labels, patch_artist=True,
        showfliers=False, whis=[5, 95],
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel(y_label)
    ax.set_title(title)

    fig.savefig(fig_dir / "fig4_jitter.pdf")
    fig.savefig(fig_dir / "fig4_jitter.png")
    plt.close(fig)
    print(f"  Saved fig4_jitter")


# =============================================================================
# 5. Detection Completeness
# =============================================================================

def analyze_detection(df_clean: pd.DataFrame, fig_dir: Path):
    """Detection completeness: how often all 12 joints are detected."""
    print("\n--- Detection Completeness ---")

    for model in MODEL_ORDER:
        m = df_clean[df_clean["model"] == model]
        full = (m["n_valid_joints"] == 12).sum()
        pct = full / len(m) * 100
        print(f"  {model:12s}: {pct:.1f}% full detection (12/12)")

    # Error by n_valid_joints
    fig, ax = plt.subplots()
    for model in MODEL_ORDER:
        m = df_clean[df_clean["model"] == model]
        grouped = m.groupby("n_valid_joints")["nmpjpe"].mean()
        ax.plot(
            grouped.index, grouped.values,
            "o-", color=MODEL_COLORS[model], label=model, markersize=4,
        )

    ax.set_xlabel("Number of Valid Joints (out of 12)")
    ax.set_ylabel("Mean NMPJPE (%)")
    ax.set_title("Accuracy vs Detection Completeness")
    ax.legend()

    fig.savefig(fig_dir / "fig5_detection.pdf")
    fig.savefig(fig_dir / "fig5_detection.png")
    plt.close(fig)
    print(f"  Saved fig5_detection")


# =============================================================================
# 6. Per-Joint Analysis
# =============================================================================

def analyze_per_joint(df: pd.DataFrame, fig_dir: Path):
    """Per-joint NMPJPE heatmap."""
    print("\n--- Per-Joint Analysis ---")

    joint_names_short = [
        "L.Shoulder", "R.Shoulder", "L.Elbow", "R.Elbow",
        "L.Wrist", "R.Wrist", "L.Hip", "R.Hip",
        "L.Knee", "R.Knee", "L.Ankle", "R.Ankle",
    ]

    matrix = np.zeros((len(MODEL_ORDER), 12))
    for i, model in enumerate(MODEL_ORDER):
        m = df[df["model"] == model]
        for j, jname in enumerate(COMPARABLE_JOINT_NAMES):
            col = f"error_{jname}"
            if col in m.columns:
                matrix[i, j] = m[col].mean()

    fig, ax = plt.subplots(figsize=(9, 3))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(joint_names_short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)

    for i in range(len(MODEL_ORDER)):
        for j in range(12):
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, label="NMPJPE (%)", shrink=0.8)
    ax.set_title("Per-Joint Error")

    fig.savefig(fig_dir / "fig6_perjoint.pdf")
    fig.savefig(fig_dir / "fig6_perjoint.png")
    plt.close(fig)
    print(f"  Saved fig6_perjoint")


# =============================================================================
# 7. Trade-off Radar
# =============================================================================

def analyze_tradeoff(df: pd.DataFrame, jitter_dir: Path, fig_dir: Path):
    """Multi-metric radar chart for trade-off overview."""
    print("\n--- Trade-off Overview ---")

    # Collect metrics (normalize to 0-1 where 1 = best)
    metrics = {}
    for model in MODEL_ORDER:
        m = df[df["model"] == model]
        metrics[model] = {
            "accuracy": m["nmpjpe"].mean(),
            "detection": (m["n_valid_joints"] == 12).mean() * 100,
            "rotation_robustness": 0,  # filled below
            "multi_person_robustness": 0,
        }
        # Rotation: lower degradation = better
        frontal = m[m["rotation_bin"] <= 20]["nmpjpe"].mean()
        lateral = m[m["rotation_bin"] >= 60]["nmpjpe"].mean()
        metrics[model]["rotation_robustness"] = (lateral / frontal - 1) * 100 if frontal > 0 else 0

        # Multi-person
        clean_m = m[~m["is_multi_person"]]["nmpjpe"].mean()
        multi_m = m[m["is_multi_person"]]["nmpjpe"].mean()
        metrics[model]["multi_person_robustness"] = (
            (multi_m / clean_m - 1) * 100 if clean_m > 0 and not np.isnan(multi_m) else 0
        )

    # Add jitter
    jitter_path = jitter_dir / "jitter_results.csv"
    if jitter_path.exists():
        jdf = pd.read_csv(jitter_path)
        jitter_col = "keypoint_jitter" if "keypoint_jitter" in jdf.columns else "nmpjpe_jitter"
        for model in MODEL_ORDER:
            metrics[model]["jitter"] = jdf[jdf["model"] == model][jitter_col].mean()
    else:
        for model in MODEL_ORDER:
            metrics[model]["jitter"] = 0

    # Print table
    print(f"  {'Model':12s} {'Accuracy':>10s} {'Detection':>10s} {'Rot.Deg':>10s} {'MP.Deg':>10s} {'Jitter':>10s}")
    for model in MODEL_ORDER:
        m = metrics[model]
        print(
            f"  {model:12s} {m['accuracy']:10.1f}% {m['detection']:9.1f}% "
            f"{m['rotation_robustness']:+9.0f}% {m['multi_person_robustness']:+9.0f}% "
            f"{m['jitter']:10.2f}%"
        )

    # Radar chart
    categories = ["Accuracy\n(lower=better)", "Detection\n(higher=better)",
                   "Rotation\nRobustness", "Temporal\nStability",
                   "Multi-Person\nRobustness"]

    # Normalize: for each dimension, scale so best model = 1.0
    raw = {}
    for model in MODEL_ORDER:
        m = metrics[model]
        raw[model] = [
            1 / m["accuracy"] if m["accuracy"] > 0 else 0,  # lower accuracy = better, so invert
            m["detection"] / 100,
            1 / (1 + m["rotation_robustness"] / 100),  # lower degradation = better
            1 / (1 + m["jitter"]) if m["jitter"] > 0 else 1,  # lower jitter = better
            1 / (1 + m["multi_person_robustness"] / 100),  # lower degradation = better
        ]

    # Scale each dimension to [0, 1]
    for dim in range(5):
        vals = [raw[m][dim] for m in MODEL_ORDER]
        max_val = max(vals) if max(vals) > 0 else 1
        for model in MODEL_ORDER:
            raw[model][dim] /= max_val

    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for model in MODEL_ORDER:
        values = raw[model] + raw[model][:1]
        ax.plot(angles, values, "o-", color=MODEL_COLORS[model], label=model, markersize=4)
        ax.fill(angles, values, color=MODEL_COLORS[model], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("Model Trade-off Overview", y=1.08)

    fig.savefig(fig_dir / "fig7_tradeoff.pdf")
    fig.savefig(fig_dir / "fig7_tradeoff.png")
    plt.close(fig)
    print(f"  Saved fig7_tradeoff")


# =============================================================================
# Summary Statistics Export
# =============================================================================

def export_summary(df: pd.DataFrame, df_clean: pd.DataFrame, results_dir: Path):
    """Export summary statistics as JSON for thesis reference."""
    summary = {"models": {}}

    for model in MODEL_ORDER:
        valid = df[df["model"] == model]
        clean = df_clean[df_clean["model"] == model]

        single = valid[~valid["is_multi_person"]]
        multi = valid[valid["is_multi_person"]]

        summary["models"][model] = {
            "accuracy": {
                "mean_nmpjpe": round(valid["nmpjpe"].mean(), 2),
                "std_nmpjpe": round(valid["nmpjpe"].std(), 2),
                "median_nmpjpe": round(valid["nmpjpe"].median(), 2),
                "n_frames": len(valid),
            },
            "detection": {
                "full_detection_rate": round((clean["n_valid_joints"] == 12).mean() * 100, 1),
                "detection_failure_rate": round(
                    df[df["model"] == model]["is_detection_failure"].mean() * 100, 2
                ),
            },
            "multi_person": {
                "single_mean": round(single["nmpjpe"].mean(), 2) if len(single) > 0 else None,
                "multi_mean": round(multi["nmpjpe"].mean(), 2) if len(multi) > 0 else None,
                "n_multi_frames": len(multi),
            },
        }

    summary["total_frames"] = len(df)
    summary["total_valid_frames"] = len(df[~df["is_detection_failure"] & ~df["is_outlier"]])
    summary["n_videos"] = df.groupby(["video_id", "exercise", "camera"]).ngroups

    out_path = results_dir / "summary_statistics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "evaluation_v2" / "results" / "rehab24" / "frame_results.csv",
        help="Path to frame_results.csv",
    )
    args = parser.parse_args()

    results_dir = args.results.parent
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_all, df_clean, df_valid = load_and_filter(args.results)

    # Run all analyses on valid data (no detection failures, no outliers)
    analyze_accuracy(df_valid, fig_dir)
    analyze_rotation(df_valid, fig_dir)
    analyze_multi_person(df_valid, fig_dir)
    analyze_jitter(results_dir, fig_dir)
    analyze_detection(df_clean, fig_dir)
    analyze_per_joint(df_valid, fig_dir)
    analyze_tradeoff(df_valid, results_dir, fig_dir)
    export_summary(df_valid, df_clean, results_dir)

    print("\n=== Analysis complete ===")


if __name__ == "__main__":
    main()
