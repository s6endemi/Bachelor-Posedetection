"""
Generate the five new Chapter 5 figures for the thesis.

Figures produced (saved to thesis/fig/ as PDF + PNG):
  1. fig_accuracy_boxplot            — Section 5.2
  2. fig_speed_accuracy_scatter      — Section 5.3/5.8
  3. fig_cross_dataset_slopegraph    — Section 5.4
  4. fig_rotation_line               — Section 5.5
  5. fig_jitter_boxplot              — Section 5.7

Design choices following reviewer QA:
  - Colour convention matches the existing Coach comparison figure:
      MediaPipe = green, MoveNet = red, YOLOv8 = blue.
    Variants use three brightness shades per family.
  - Plot titles are removed; methodological context goes into LaTeX captions.
  - Axis labels are kept short; "valid subset", "CPU", "subject-cluster",
    "bin centre" etc. belong in the captions, not on the axes.
  - Speed-accuracy scatter uses per-variant label offsets and no legend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Paths and configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "evaluation_v2" / "results"
OUT_DIR = PROJECT_ROOT / "thesis" / "fig"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Main-model colours consistent with the Coach comparison figure.
FAMILY_COLOR = {
    "MediaPipe": "#2E7D32",   # green  (Coach figure uses green)
    "MoveNet":   "#C62828",   # red    (Coach figure uses red)
    "YOLOv8":    "#1565C0",   # blue   (Coach figure uses blue)
}

MAIN_MODELS = ["MediaPipe", "MoveNet", "YOLOv8"]

# Variant palette: three brightness shades per family.
VARIANT_COLORS = {
    # MediaPipe family — greens
    "MediaPipe_Lite":         "#A5D6A7",  # light
    "MediaPipe_Full":         "#2E7D32",  # main
    "MediaPipe_Heavy":        "#1B5E20",  # dark
    # MoveNet family — reds
    "MoveNet_SP_Lightning":   "#EF9A9A",  # light
    "MoveNet_SP_Thunder":     "#E57373",  # medium
    "MoveNet_MultiPose":      "#C62828",  # main
    # YOLOv8 family — blues
    "YOLOv8_Nano":            "#90CAF9",  # light
    "YOLOv8_Small":           "#42A5F5",  # medium
    "YOLOv8_Medium":          "#1565C0",  # dark
}

VARIANT_LABEL = {
    "MediaPipe_Lite":       "MediaPipe Lite",
    "MediaPipe_Full":       "MediaPipe Full",
    "MediaPipe_Heavy":      "MediaPipe Heavy",
    "MoveNet_SP_Lightning": "MoveNet SP Lightning",
    "MoveNet_SP_Thunder":   "MoveNet SP Thunder",
    "MoveNet_MultiPose":    "MoveNet MultiPose",
    "YOLOv8_Nano":          "YOLOv8 Nano",
    "YOLOv8_Small":         "YOLOv8 Small",
    "YOLOv8_Medium":        "YOLOv8 Medium",
}

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":    150,
    "savefig.dpi":   300,
})


def save(fig, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Figure 1 — Accuracy Boxplot (Section 5.2)
# =============================================================================

def fig_accuracy_boxplot() -> None:
    df = pd.read_csv(RESULTS / "rehab24" / "patient_level_aggregates.csv")
    df = df[df["model"].isin(MAIN_MODELS)].copy()

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    data = [df.loc[df["model"] == m, "patient_mean_video_median_nmpjpe"].values
            for m in MAIN_MODELS]

    bp = ax.boxplot(
        data, tick_labels=MAIN_MODELS, patch_artist=True,
        widths=0.55, showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
    )
    for patch, m in zip(bp["boxes"], MAIN_MODELS):
        patch.set_facecolor(FAMILY_COLOR[m])
        patch.set_alpha(0.30)
        patch.set_edgecolor(FAMILY_COLOR[m])

    rng = np.random.default_rng(seed=0)
    for i, m in enumerate(MAIN_MODELS, start=1):
        vals = df.loc[df["model"] == m, "patient_mean_video_median_nmpjpe"].values
        x = rng.normal(loc=i, scale=0.06, size=len(vals))
        ax.scatter(x, vals, s=22, color=FAMILY_COLOR[m],
                   edgecolor="black", linewidth=0.6, zorder=3)

    ax.set_ylabel("Mean NMPJPE (%)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    save(fig, "fig_accuracy_boxplot")


# =============================================================================
# Figure 2 — Speed-Accuracy Scatter (Section 5.3 / 5.8)
# =============================================================================

LABEL_OFFSETS = {
    # Per-variant offsets and alignment to avoid overlap.
    # Format: (dx, dy, ha). Clusters:
    #   Low FPS (< 12): YOLOv8 Medium, MediaPipe Heavy, YOLOv8 Small
    #   Mid FPS (12-25): MediaPipe Full, YOLOv8 Nano, MediaPipe Lite
    #   High MultiPose (~28): MoveNet MultiPose, MoveNet SP Thunder
    #   Far-right (~100): MoveNet SP Lightning
    "YOLOv8_Medium":          (8,  4,  "left"),     # far left, label right
    "MediaPipe_Heavy":        (-8, 4,  "right"),    # label left
    "YOLOv8_Small":           (8, -11, "left"),     # label below-right
    "MediaPipe_Full":         (8, -11, "left"),     # label below-right
    "YOLOv8_Nano":            (8,  8,  "left"),     # label above-right
    "MediaPipe_Lite":         (-8, 4,  "right"),    # label left (avoid MoveNet SP Thunder)
    "MoveNet_MultiPose":      (8, -11, "left"),     # bottom centre, label below-right
    "MoveNet_SP_Thunder":     (8,  8,  "left"),     # label above-right
    "MoveNet_SP_Lightning":   (-10, 4, "right"),    # far right, label left
}


def fig_speed_accuracy_scatter() -> None:
    frame = pd.read_csv(RESULTS / "rehab24_all" / "frame_results_all.csv")
    bench = pd.read_csv(RESULTS / "inference_benchmark_all.csv")

    valid = frame[
        (~frame["is_detection_failure"])
        & (~frame["is_outlier"])
        & (frame["mocap_erroneous"] == 0)
    ]
    acc = valid.groupby("model")["nmpjpe"].mean().rename("mean_nmpjpe")
    merged = bench.set_index("model").join(acc, how="inner").reset_index()

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    # Pareto frontier (computed before points to sit below)
    sorted_pts = merged.sort_values("fps", ascending=False)
    pareto_x, pareto_y = [], []
    best = float("inf")
    for _, row in sorted_pts.iterrows():
        if row["mean_nmpjpe"] < best:
            best = row["mean_nmpjpe"]
            pareto_x.append(row["fps"])
            pareto_y.append(row["mean_nmpjpe"])
    ax.plot(pareto_x, pareto_y, "--", color="#888888", alpha=0.7,
            linewidth=1.0, zorder=1)

    for _, row in merged.iterrows():
        m = row["model"]
        color = VARIANT_COLORS.get(m, "#444444")
        ax.scatter(row["fps"], row["mean_nmpjpe"], s=80,
                   color=color, edgecolor="black", linewidth=0.7, zorder=3)

        dx, dy, ha = LABEL_OFFSETS.get(m, (6, 4, "left"))
        ax.annotate(VARIANT_LABEL.get(m, m),
                    xy=(row["fps"], row["mean_nmpjpe"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, color=color, ha=ha)

    ax.set_xlabel("Throughput (FPS)")
    ax.set_ylabel("Mean NMPJPE (%)")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    save(fig, "fig_speed_accuracy_scatter")


# =============================================================================
# Figure 3 — Cross-Dataset Slopegraph (Section 5.4)
# =============================================================================

def fig_cross_dataset_slopegraph() -> None:
    rehab = pd.read_csv(RESULTS / "rehab24_all" / "frame_results_all.csv")
    coco = pd.read_csv(RESULTS / "coco" / "coco_results.csv")

    name_map = {
        "MediaPipe":       "MediaPipe_Full",
        "MoveNet":         "MoveNet_MultiPose",
        "YOLOv8":          "YOLOv8_Nano",
        "MediaPipe_Lite":  "MediaPipe_Lite",
        "MediaPipe_Heavy": "MediaPipe_Heavy",
        "MoveNet_SP_Lightning": "MoveNet_SP_Lightning",
        "MoveNet_SP_Thunder":   "MoveNet_SP_Thunder",
        "YOLOv8s":         "YOLOv8_Small",
        "YOLOv8m":         "YOLOv8_Medium",
    }
    coco["model_norm"] = coco["model"].map(name_map).fillna(coco["model"])

    r_valid = rehab[(~rehab["is_detection_failure"]) & (~rehab["is_outlier"])
                     & (rehab["mocap_erroneous"] == 0)]
    c_valid = coco[(~coco["is_detection_failure"]) & (~coco["is_outlier"])]

    r_mean = r_valid.groupby("model")["nmpjpe"].mean().to_dict()
    c_mean = c_valid.groupby("model_norm")["nmpjpe"].mean().to_dict()

    variants = [m for m in VARIANT_COLORS if m in r_mean and m in c_mean]
    r_order = sorted(variants, key=lambda m: r_mean[m])
    c_order = sorted(variants, key=lambda m: c_mean[m])
    r_rank = {m: r_order.index(m) + 1 for m in variants}
    c_rank = {m: c_order.index(m) + 1 for m in variants}

    fig, ax = plt.subplots(figsize=(5.8, 4.4))

    for m in variants:
        color = VARIANT_COLORS[m]
        ax.plot([0, 1], [r_rank[m], c_rank[m]], "-", color=color,
                linewidth=1.6, alpha=0.85, zorder=2)
        ax.scatter([0, 1], [r_rank[m], c_rank[m]], s=50, color=color,
                   edgecolor="black", linewidth=0.5, zorder=3)
        # Full label only on the right side
        ax.annotate(VARIANT_LABEL.get(m, m), xy=(1.03, c_rank[m]),
                    fontsize=7.5, color=color, va="center")
        # Compact rank number on the left
        ax.annotate(f"{r_rank[m]}", xy=(-0.04, r_rank[m]),
                    fontsize=8, color=color, va="center", ha="right",
                    fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["REHAB24-6", "COCO val2017"])
    ax.set_ylabel("Rank (1 = lowest NMPJPE)")
    ax.invert_yaxis()
    ax.set_xlim(-0.2, 1.6)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    save(fig, "fig_cross_dataset_slopegraph")


# =============================================================================
# Figure 4 — Rotation Line Plot (Section 5.5)
# =============================================================================

def fig_rotation_line() -> None:
    df = pd.read_csv(RESULTS / "rehab24" / "frame_results.csv")
    valid = df[(~df["is_detection_failure"]) & (~df["is_outlier"])
                & (df["mocap_erroneous"] == 0)]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))

    for m in MAIN_MODELS:
        sub = valid[valid["model"] == m]
        grouped = sub.groupby("rotation_bin")["nmpjpe"].mean().sort_index()
        ax.plot(grouped.index + 5, grouped.values, "o-",
                color=FAMILY_COLOR[m], label=m, linewidth=1.8,
                markersize=5, markeredgecolor="black", markeredgewidth=0.4)

    ax.set_xlabel("Rotation bin centre (°)")
    ax.set_ylabel("Mean NMPJPE (%)")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)

    save(fig, "fig_rotation_line")


# =============================================================================
# Figure 5 — Jitter Boxplot (Section 5.7)
# =============================================================================

def fig_jitter_boxplot() -> None:
    df = pd.read_csv(RESULTS / "rehab24" / "jitter_results.csv")
    df = df[df["model"].isin(MAIN_MODELS)].copy()

    fig, ax = plt.subplots(figsize=(5.4, 3.4))

    data = [df.loc[df["model"] == m, "keypoint_jitter"].values
            for m in MAIN_MODELS]
    bp = ax.boxplot(
        data, tick_labels=MAIN_MODELS, patch_artist=True,
        widths=0.55, showfliers=False, whis=[5, 95],
        medianprops=dict(color="black", linewidth=1.2),
    )
    for patch, m in zip(bp["boxes"], MAIN_MODELS):
        patch.set_facecolor(FAMILY_COLOR[m])
        patch.set_alpha(0.30)
        patch.set_edgecolor(FAMILY_COLOR[m])

    ax.set_ylabel("Keypoint displacement (%)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    save(fig, "fig_jitter_boxplot")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("Generating Chapter 5 figures from canonical artifacts...")
    print()

    print("1/5  Accuracy boxplot (Section 5.2)")
    fig_accuracy_boxplot()

    print("2/5  Speed-accuracy scatter (Section 5.3/5.8)")
    fig_speed_accuracy_scatter()

    print("3/5  Cross-dataset slopegraph (Section 5.4)")
    fig_cross_dataset_slopegraph()

    print("4/5  Rotation line plot (Section 5.5)")
    fig_rotation_line()

    print("5/5  Jitter boxplot (Section 5.7)")
    fig_jitter_boxplot()

    print()
    print(f"All five figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
