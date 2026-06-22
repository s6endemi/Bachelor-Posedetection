"""
Per-exercise breakdown of accuracy and jitter on REHAB24-6.

Aggregates mean NMPJPE (valid subset) and mean frame-to-frame jitter per
exercise x model. Produces a heatmap to visualize whether model rankings are
stable across exercise types or vary by movement.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_v2.result_utils import MAIN_MODELS, filter_valid, load_results


def per_exercise_nmpjpe(results_path: Path) -> pd.DataFrame:
    df = load_results(results_path)
    valid = filter_valid(df)
    valid = valid[valid["model"].isin(MAIN_MODELS)]
    agg = (
        valid.groupby(["exercise", "model"])
        .agg(
            n=("nmpjpe", "size"),
            mean_nmpjpe=("nmpjpe", "mean"),
            median_nmpjpe=("nmpjpe", "median"),
            std_nmpjpe=("nmpjpe", "std"),
        )
        .reset_index()
    )
    return agg


def per_exercise_jitter(jitter_path: Path) -> pd.DataFrame:
    df = pd.read_csv(jitter_path)
    df = df[df["model"].isin(MAIN_MODELS)]
    agg = (
        df.groupby(["exercise", "model"])
        .agg(
            n_pairs=("keypoint_jitter", "size"),
            mean_jitter=("keypoint_jitter", "mean"),
            median_jitter=("keypoint_jitter", "median"),
        )
        .reset_index()
    )
    return agg


def compute_rankings(agg: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """For each exercise, rank models from best (1) to worst (3)."""
    rows = []
    for exercise, grp in agg.groupby("exercise"):
        sorted_grp = grp.sort_values(metric_col).reset_index(drop=True)
        for rk, (_, row) in enumerate(sorted_grp.iterrows(), start=1):
            rows.append({
                "exercise": exercise,
                "model": row["model"],
                "rank": rk,
                "value": row[metric_col],
            })
    return pd.DataFrame(rows)


def make_heatmap(agg: pd.DataFrame, value_col: str, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    pivot = agg.pivot(index="exercise", columns="model", values=value_col)
    pivot = pivot[MAIN_MODELS]
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > pivot.values.mean() else "black",
                    fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label(value_col)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def print_summary(nm_agg: pd.DataFrame, jit_agg: pd.DataFrame) -> None:
    print("\n=== Mean NMPJPE (%) by exercise x model, valid subset ===")
    pivot = nm_agg.pivot(index="exercise", columns="model", values="mean_nmpjpe")[MAIN_MODELS]
    print(pivot.round(2).to_string())

    rk = compute_rankings(nm_agg, "mean_nmpjpe")
    print("\n=== NMPJPE rank by exercise (1=best) ===")
    rk_pivot = rk.pivot(index="exercise", columns="model", values="rank")[MAIN_MODELS]
    print(rk_pivot.to_string())

    print("\n=== Mean jitter (%) by exercise x model ===")
    jit_pivot = jit_agg.pivot(index="exercise", columns="model", values="mean_jitter")[MAIN_MODELS]
    print(jit_pivot.round(2).to_string())

    jit_rk = compute_rankings(jit_agg, "mean_jitter")
    print("\n=== Jitter rank by exercise (1=lowest jitter) ===")
    jit_rk_pivot = jit_rk.pivot(index="exercise", columns="model", values="rank")[MAIN_MODELS]
    print(jit_rk_pivot.to_string())


def main() -> None:
    results_path = PROJECT_ROOT / "evaluation_v2" / "results" / "rehab24" / "frame_results.csv"
    jitter_path = PROJECT_ROOT / "evaluation_v2" / "results" / "rehab24" / "jitter_results.csv"
    out_dir = PROJECT_ROOT / "evaluation_v2" / "results"

    nm_agg = per_exercise_nmpjpe(results_path)
    jit_agg = per_exercise_jitter(jitter_path)

    nm_agg.to_csv(out_dir / "per_exercise_nmpjpe.csv", index=False)
    jit_agg.to_csv(out_dir / "per_exercise_jitter.csv", index=False)

    make_heatmap(nm_agg, "mean_nmpjpe",
                 "Mean NMPJPE (%) by exercise x model",
                 out_dir / "per_exercise_nmpjpe_heatmap.png")
    make_heatmap(jit_agg, "mean_jitter",
                 "Mean jitter (%) by exercise x model",
                 out_dir / "per_exercise_jitter_heatmap.png")

    print_summary(nm_agg, jit_agg)
    print(f"\nSaved CSV+plots to: {out_dir}")


if __name__ == "__main__":
    main()
