"""
Failure-mode taxonomy for REHAB24-6 frame-level results.

Classifies every failure frame into one of four mutually exclusive categories
and reports the per-model distribution, plus a sensitivity analysis over the
confidence-collapse threshold.

Categories (applied in order, mutually exclusive):

1. Missing-Detection   : is_detection_failure == True
                        (model returned no detection at all)
2. Confidence-Collapse : is_outlier AND n_valid_joints < COLLAPSE_THRESHOLD
                        (detection succeeded but most keypoints fell below
                         the confidence filter, so NMPJPE is computed on a
                         small, noisy subset)
3. Multi-Person-Confusion : is_outlier AND n_valid_joints >= threshold
                           AND extra_person_severity >= 2
                           (outlier in a frame with visible secondary persons --
                            selection stage likely grabbed the wrong subject)
4. Keypoint-Displacement : is_outlier AND n_valid_joints >= threshold
                           AND extra_person_severity < 2
                           (outlier in a single-person frame with sufficient
                            valid joints -- pure catastrophic misprediction)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_v2.result_utils import MAIN_MODELS, load_results

CATEGORY_ORDER = [
    "Missing-Detection",
    "Confidence-Collapse",
    "Multi-Person-Confusion",
    "Keypoint-Displacement",
]
COLLAPSE_THRESHOLD_DEFAULT = 6
SENSITIVITY_THRESHOLDS = [4, 6, 8]


def classify(df: pd.DataFrame, collapse_threshold: int) -> pd.DataFrame:
    """Add a 'failure_category' column. Non-failures get 'OK'."""
    out = df.copy()
    cat = pd.Series("OK", index=out.index)

    is_outlier = out["is_outlier"] == True  # noqa: E712
    is_det_fail = out["is_detection_failure"] == True  # noqa: E712
    low_joints = out["n_valid_joints"] < collapse_threshold
    multi = out["extra_person_severity"] >= 2

    cat[is_det_fail] = "Missing-Detection"
    cat[is_outlier & low_joints & ~is_det_fail] = "Confidence-Collapse"
    cat[is_outlier & ~low_joints & multi & ~is_det_fail] = "Multi-Person-Confusion"
    cat[is_outlier & ~low_joints & ~multi & ~is_det_fail] = "Keypoint-Displacement"

    out["failure_category"] = cat.values
    return out


def per_model_counts(df: pd.DataFrame) -> pd.DataFrame:
    totals = df.groupby("model").size().rename("n_frames")
    counts = (
        df[df["failure_category"] != "OK"]
        .groupby(["model", "failure_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORY_ORDER, fill_value=0)
    )
    counts = counts.join(totals)
    counts["n_failure"] = counts[CATEGORY_ORDER].sum(axis=1)
    counts["failure_rate_pct"] = counts["n_failure"] / counts["n_frames"] * 100
    for cat in CATEGORY_ORDER:
        counts[f"{cat}_pct_of_failures"] = counts[cat] / counts["n_failure"] * 100
        counts[f"{cat}_pct_of_frames"] = counts[cat] / counts["n_frames"] * 100
    return counts


def sensitivity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for thr in SENSITIVITY_THRESHOLDS:
        classified = classify(df, thr)
        subset = classified[classified["model"].isin(MAIN_MODELS)]
        for model in MAIN_MODELS:
            model_df = subset[subset["model"] == model]
            n_fail = int((model_df["failure_category"] != "OK").sum())
            row = {"threshold": thr, "model": model, "n_failure": n_fail}
            for cat in CATEGORY_ORDER:
                n_cat = int((model_df["failure_category"] == cat).sum())
                row[f"{cat}"] = n_cat
                row[f"{cat}_pct_of_failures"] = 0.0 if n_fail == 0 else n_cat / n_fail * 100
            rows.append(row)
    return pd.DataFrame(rows)


def make_plot(counts: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    models = [m for m in MAIN_MODELS if m in counts.index]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(models))
    bottom = np.zeros(len(models))
    palette = {
        "Missing-Detection": "#9e9e9e",
        "Confidence-Collapse": "#2b7bba",
        "Multi-Person-Confusion": "#d98c00",
        "Keypoint-Displacement": "#c1332d",
    }
    for cat in CATEGORY_ORDER:
        vals = np.array([counts.loc[m, cat] for m in models], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=cat, color=palette[cat], edgecolor="white", linewidth=0.6)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Failure frames")
    ax.set_title("Failure-mode taxonomy by model (REHAB24-6)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def write_outputs(
    classified: pd.DataFrame,
    counts: pd.DataFrame,
    sens: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "failure_mode_summary.csv"
    json_path = out_dir / "failure_mode_summary.json"
    per_video_path = out_dir / "failure_mode_per_video.csv"
    sens_path = out_dir / "failure_mode_sensitivity.csv"

    counts.reset_index().to_csv(csv_path, index=False)
    sens.to_csv(sens_path, index=False)

    per_video = (
        classified[classified["failure_category"] != "OK"]
        .groupby(["video_id", "model", "failure_category"])
        .size()
        .reset_index(name="n")
    )
    per_video.to_csv(per_video_path, index=False)

    payload = {
        "collapse_threshold_default": COLLAPSE_THRESHOLD_DEFAULT,
        "sensitivity_thresholds": SENSITIVITY_THRESHOLDS,
        "categories": CATEGORY_ORDER,
        "per_model_summary": counts.reset_index().to_dict(orient="records"),
        "sensitivity": sens.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {per_video_path}")
    print(f"Saved: {sens_path}")


def print_summary(counts: pd.DataFrame) -> None:
    print("\n| Model | n | Failures | Miss-Det | Conf-Collapse | Multi-Pers | Kp-Displ |")
    print("|-------|---:|---------:|---------:|--------------:|-----------:|---------:|")
    for model in MAIN_MODELS:
        if model not in counts.index:
            continue
        row = counts.loc[model]
        n = int(row["n_frames"])
        nf = int(row["n_failure"])
        md = int(row["Missing-Detection"])
        cc = int(row["Confidence-Collapse"])
        mp = int(row["Multi-Person-Confusion"])
        kd = int(row["Keypoint-Displacement"])
        print(
            f"| {model} | {n:,} | {nf:,} ({nf/n*100:.2f}%) | "
            f"{md:,} ({md/nf*100:.1f}%) | {cc:,} ({cc/nf*100:.1f}%) | "
            f"{mp:,} ({mp/nf*100:.1f}%) | {kd:,} ({kd/nf*100:.1f}%) |"
        )


def main() -> None:
    results_path = PROJECT_ROOT / "evaluation_v2" / "results" / "rehab24" / "frame_results.csv"
    df = load_results(results_path)
    df = df[df["model"].isin(MAIN_MODELS)].copy()

    classified = classify(df, COLLAPSE_THRESHOLD_DEFAULT)
    counts = per_model_counts(classified)
    sens = sensitivity_analysis(df)

    out_dir = PROJECT_ROOT / "evaluation_v2" / "results"
    write_outputs(classified, counts, sens, out_dir)

    print_summary(counts)
    print("\nSensitivity over COLLAPSE_THRESHOLD:")
    pivot = sens.pivot(index="model", columns="threshold", values="Confidence-Collapse_pct_of_failures")
    print(pivot.round(1).to_string())

    make_plot(counts, out_dir / "failure_mode_taxonomy.png")
    print(f"\nPlot saved to: {out_dir / 'failure_mode_taxonomy.png'}")


if __name__ == "__main__":
    main()
