"""
Reproducible hip-sensitivity analysis for REHAB24-6 result files.

Reports mean error for:
- all 12 comparable joints
- hips only
- without hips (10 joints)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_v2.config import COMPARABLE_JOINT_NAMES
from evaluation_v2.result_utils import (
    MAIN_MODELS,
    MAIN_VARIANT_MODELS,
    filter_main_models,
    filter_valid,
    load_results,
)


HIP_JOINTS = ["left_hip", "right_hip"]
NON_HIP_JOINTS = [name for name in COMPARABLE_JOINT_NAMES if name not in HIP_JOINTS]


def _per_row_mean(df: pd.DataFrame, joint_names: list[str]) -> pd.Series:
    cols = [f"error_{name}" for name in joint_names]
    return df[cols].mean(axis=1, skipna=True)


def build_summary(results_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_results(results_path)
    valid = filter_valid(df)

    all_mean = valid.groupby("model")["nmpjpe"].mean().rename("all_12_mean")
    hips_mean = _per_row_mean(valid, HIP_JOINTS).groupby(valid["model"]).mean().rename("hips_only_mean")
    non_hips_mean = (
        _per_row_mean(valid, NON_HIP_JOINTS)
        .groupby(valid["model"])
        .mean()
        .rename("without_hips_mean")
    )

    summary = pd.concat([all_mean, hips_mean, non_hips_mean], axis=1).reset_index()
    summary["hip_minus_all"] = summary["hips_only_mean"] - summary["all_12_mean"]
    summary["all_minus_without_hips"] = summary["all_12_mean"] - summary["without_hips_mean"]
    summary = summary.sort_values("all_12_mean").reset_index(drop=True)

    main_names = set(MAIN_MODELS) | set(MAIN_VARIANT_MODELS)
    main_summary = summary[summary["model"].isin(main_names)].copy()

    if {"MediaPipe", "MoveNet"}.issubset(set(main_summary["model"])):
        mp = main_summary.loc[main_summary["model"] == "MediaPipe"].iloc[0]
        mn = main_summary.loc[main_summary["model"] == "MoveNet"].iloc[0]
        gap_row = pd.DataFrame(
            [
                {
                    "model": "MediaPipe_minus_MoveNet_gap",
                    "all_12_mean": mp["all_12_mean"] - mn["all_12_mean"],
                    "hips_only_mean": mp["hips_only_mean"] - mn["hips_only_mean"],
                    "without_hips_mean": mp["without_hips_mean"] - mn["without_hips_mean"],
                    "hip_minus_all": (mp["hips_only_mean"] - mn["hips_only_mean"]) - (mp["all_12_mean"] - mn["all_12_mean"]),
                    "all_minus_without_hips": (mp["all_12_mean"] - mn["all_12_mean"]) - (mp["without_hips_mean"] - mn["without_hips_mean"]),
                }
            ]
        )
        main_summary = pd.concat([main_summary, gap_row], ignore_index=True)

    return summary, main_summary


def write_outputs(summary: pd.DataFrame, main_summary: pd.DataFrame, results_path: Path) -> None:
    out_dir = PROJECT_ROOT / "evaluation_v2" / "results"
    csv_path = out_dir / "hip_sensitivity_summary.csv"
    json_path = out_dir / "hip_sensitivity_summary.json"

    summary.to_csv(csv_path, index=False)
    payload = {
        "source_results": str(results_path),
        "subset": "valid = no detection failure, no outlier, no mocap error",
        "models_all_variants": summary.to_dict(orient="records"),
        "main_models_and_gap": main_summary.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print()
    print("| Model | All 12 Joints | Hips Only | Without Hips |")
    print("|-------|---------------:|----------:|-------------:|")
    for _, row in main_summary.iterrows():
        print(
            f"| {row['model']} | {row['all_12_mean']:.2f}% | "
            f"{row['hips_only_mean']:.2f}% | {row['without_hips_mean']:.2f}% |"
        )


def main() -> None:
    results_path = PROJECT_ROOT / "evaluation_v2" / "results" / "rehab24_all" / "frame_results_all.csv"
    summary, main_summary = build_summary(results_path)
    write_outputs(summary, main_summary, results_path)


if __name__ == "__main__":
    main()
