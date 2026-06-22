"""
Canonical paired statistics for the main REHAB24-6 model comparison.

Primary inference unit:
- segmentation person_id cluster

Aggregation strategy:
1. compute per-video median NMPJPE
2. average those video medians within each segmentation person_id cluster
3. run paired bootstrap / paired sign-flip permutation on cluster-level deltas

Video-level aggregates are exported as a secondary sensitivity analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_v2.result_utils import MAIN_MODELS, filter_main_models, filter_valid, load_results


RNG_SEED = 20260412


@dataclass
class PairwiseResult:
    model_a: str
    model_b: str
    n_pairs: int
    mean_diff: float
    median_diff: float
    effect_size_dz: float
    ci_low: float
    ci_high: float
    p_value: float
    p_value_holm: float | None = None
    level: str = "patient"


def build_video_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["patient_id", "video_id", "exercise", "camera", "model"], as_index=False)["nmpjpe"]
        .median()
        .rename(columns={"nmpjpe": "video_median_nmpjpe"})
    )


def build_patient_aggregates(video_df: pd.DataFrame) -> pd.DataFrame:
    return (
        video_df.groupby(["patient_id", "model"], as_index=False)["video_median_nmpjpe"]
        .mean()
        .rename(columns={"video_median_nmpjpe": "patient_mean_video_median_nmpjpe"})
    )


def paired_bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator, n_boot: int = 20000) -> tuple[float, float]:
    n = len(diffs)
    samples = rng.choice(diffs, size=(n_boot, n), replace=True)
    boot_means = samples.mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return float(ci_low), float(ci_high)


def paired_sign_flip_pvalue(
    diffs: np.ndarray,
    rng: np.random.Generator | None = None,
    n_perm: int = 50000,
    exact_max_n: int = 20,
) -> float:
    observed = abs(float(diffs.mean()))
    if len(diffs) <= exact_max_n:
        total = 0
        extreme = 0
        for signs in product((-1.0, 1.0), repeat=len(diffs)):
            total += 1
            permuted = abs(float(np.mean(np.asarray(signs) * diffs)))
            if permuted >= observed - 1e-12:
                extreme += 1
        return float(extreme / total)

    if rng is None:
        raise ValueError("rng is required for Monte Carlo sign-flip sampling")

    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(diffs)), replace=True)
    permuted = np.abs((signs * diffs).mean(axis=1))
    p_value = (np.sum(permuted >= observed) + 1) / (n_perm + 1)
    return float(p_value)


def paired_effect_size_dz(diffs: np.ndarray) -> float:
    std = float(np.std(diffs, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(diffs) / std)


def compute_pairwise_results(
    aggregate_df: pd.DataFrame,
    value_col: str,
    level: str,
    rng_seed: int = RNG_SEED,
) -> list[PairwiseResult]:
    pivot = aggregate_df.pivot(index="patient_id" if level == "patient" else ["patient_id", "video_id", "exercise", "camera"], columns="model", values=value_col)
    pivot = pivot.sort_index()
    results: list[PairwiseResult] = []

    for model_a, model_b in combinations(MAIN_MODELS, 2):
        pair_df = pivot[[model_a, model_b]].dropna()
        diffs = (pair_df[model_a] - pair_df[model_b]).to_numpy(dtype=float)
        rng = np.random.default_rng(rng_seed + len(results))
        ci_low, ci_high = paired_bootstrap_ci(diffs, rng)
        p_value = paired_sign_flip_pvalue(diffs, rng)
        results.append(
            PairwiseResult(
                model_a=model_a,
                model_b=model_b,
                n_pairs=len(diffs),
                mean_diff=float(np.mean(diffs)),
                median_diff=float(np.median(diffs)),
                effect_size_dz=paired_effect_size_dz(diffs),
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=p_value,
                level=level,
            )
        )
    apply_holm_correction(results)
    return results


def apply_holm_correction(results: list[PairwiseResult]) -> None:
    if not results:
        return
    ordered = sorted(enumerate(results), key=lambda item: item[1].p_value)
    m = len(ordered)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (idx, result) in enumerate(ordered):
        value = min(1.0, (m - rank) * result.p_value)
        running_max = max(running_max, value)
        adjusted[rank] = running_max
    for (idx, _), adj in zip(ordered, adjusted):
        results[idx].p_value_holm = float(adj)


def summarize_patient_models(patient_df: pd.DataFrame) -> pd.DataFrame:
    return (
        patient_df.groupby("model")["patient_mean_video_median_nmpjpe"]
        .agg(mean="mean", median="median", std="std", n_clusters="size")
        .reset_index()
        .sort_values("mean")
    )


def write_outputs(
    video_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    patient_results: list[PairwiseResult],
    video_results: list[PairwiseResult],
) -> None:
    out_dir = PROJECT_ROOT / "evaluation_v2" / "results" / "rehab24"
    video_agg_path = out_dir / "video_level_aggregates.csv"
    patient_agg_path = out_dir / "patient_level_aggregates.csv"
    patient_pairwise_path = out_dir / "patient_level_pairwise_stats.csv"
    video_pairwise_path = out_dir / "video_level_pairwise_stats.csv"
    summary_path = out_dir / "patient_level_summary.csv"
    json_path = out_dir / "patient_level_stats.json"

    video_df.to_csv(video_agg_path, index=False)
    patient_df.to_csv(patient_agg_path, index=False)
    pd.DataFrame([r.__dict__ for r in patient_results]).to_csv(patient_pairwise_path, index=False)
    pd.DataFrame([r.__dict__ for r in video_results]).to_csv(video_pairwise_path, index=False)
    summary_df = summarize_patient_models(patient_df)
    summary_df.to_csv(summary_path, index=False)

    payload = {
        "primary_inference_unit": "segmentation person_id cluster",
        "primary_aggregation": "mean of per-video median NMPJPE within each segmentation person_id cluster",
        "secondary_sensitivity_unit": "video",
        "models": MAIN_MODELS,
        "n_primary_clusters": int(patient_df["patient_id"].nunique()),
        "patient_level_summary": summary_df.to_dict(orient="records"),
        "patient_level_pairwise": [r.__dict__ for r in patient_results],
        "video_level_pairwise": [r.__dict__ for r in video_results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {video_agg_path}")
    print(f"Saved: {patient_agg_path}")
    print(f"Saved: {patient_pairwise_path}")
    print(f"Saved: {video_pairwise_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {json_path}")
    print()
    print("| Pair | n | Mean diff (A-B) | 95% CI | p | p Holm | dz |")
    print("|------|--:|----------------:|--------|---:|-------:|---:|")
    for result in patient_results:
        print(
            f"| {result.model_a} - {result.model_b} | {result.n_pairs} | {result.mean_diff:.3f} | "
            f"[{result.ci_low:.3f}, {result.ci_high:.3f}] | {result.p_value:.4f} | "
            f"{result.p_value_holm:.4f} | {result.effect_size_dz:.3f} |"
        )


def main() -> None:
    results_path = PROJECT_ROOT / "evaluation_v2" / "results" / "rehab24" / "frame_results.csv"
    df = load_results(results_path)
    valid = filter_main_models(filter_valid(df))
    video_df = build_video_aggregates(valid)
    patient_df = build_patient_aggregates(video_df)
    patient_results = compute_pairwise_results(
        patient_df,
        value_col="patient_mean_video_median_nmpjpe",
        level="patient",
    )
    video_results = compute_pairwise_results(
        video_df.rename(columns={"video_median_nmpjpe": "value"}),
        value_col="value",
        level="video",
    )
    write_outputs(video_df, patient_df, patient_results, video_results)


if __name__ == "__main__":
    main()
