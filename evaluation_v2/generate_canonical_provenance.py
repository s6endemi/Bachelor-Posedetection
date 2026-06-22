"""
Generate a canonical provenance report for thesis-facing result files.

This script freezes which CSV artifacts are authoritative and records the
subset definitions used for each headline result category.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_v2.result_utils import (
    RESULTS_ROOT,
    filter_clean,
    filter_coach_subset,
    filter_main_models,
    filter_non_coach_subset,
    filter_segmentation_multi_person,
    filter_valid,
    load_results,
    summarize_model_central_tendency,
)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _records(df):
    return df.to_dict(orient="records")


def build_report() -> dict:
    rehab_path = RESULTS_ROOT / "rehab24" / "frame_results.csv"
    rehab_all_path = RESULTS_ROOT / "rehab24_all" / "frame_results_all.csv"
    coco_path = RESULTS_ROOT / "coco" / "coco_results.csv"

    rehab = load_results(rehab_path)
    rehab_all = load_results(rehab_all_path)
    coco = load_results(coco_path)

    rehab_clean = filter_clean(rehab)
    rehab_valid = filter_valid(rehab)
    rehab_main_valid = filter_main_models(rehab_valid)
    rehab_coach_all = filter_coach_subset(rehab_clean)
    rehab_coach_valid = filter_coach_subset(rehab_valid)
    rehab_non_coach_all = filter_non_coach_subset(rehab_clean)
    rehab_seg_multi_valid = filter_segmentation_multi_person(rehab_valid)

    rehab_all_valid = filter_valid(rehab_all)
    coco_valid = filter_valid(coco, require_mocap_clean=False)

    coco_metadata = _read_json(RESULTS_ROOT / "coco" / "metadata.json")
    rehab_metadata = _read_json(RESULTS_ROOT / "rehab24" / "metadata.json")
    rehab_all_metadata = _read_json(RESULTS_ROOT / "rehab24_all" / "metadata.json")

    warnings = []
    if coco_metadata.get("total_rows") != len(coco):
        warnings.append(
            "evaluation_v2/results/coco/metadata.json does not match coco_results.csv row count."
        )
    metadata_models = set(coco_metadata.get("models", []))
    csv_models = set(coco["model"].unique().tolist())
    if metadata_models != csv_models:
        warnings.append(
            "evaluation_v2/results/coco/metadata.json does not match coco_results.csv model list."
        )
    if "c17_frontal_offset" not in rehab_all_metadata.get("eval_config", {}):
        warnings.append(
            "evaluation_v2/results/rehab24_all/metadata.json is missing key eval_config fields present in rehab24 metadata."
        )
    n_video_ids = int(rehab[["video_id", "exercise"]].drop_duplicates().shape[0])
    n_person_clusters = int(pd.Series(rehab["patient_id"]).dropna().nunique())
    if n_person_clusters and n_person_clusters != n_video_ids:
        warnings.append(
            f"Segmentation metadata maps {n_video_ids} video/exercise pairs to {n_person_clusters} person_id clusters; thesis dataset-size claims should be reconciled before final writing."
        )

    return {
        "generated_from": {
            "project_root": str(PROJECT_ROOT),
            "rehab24_csv": str(rehab_path),
            "rehab24_all_csv": str(rehab_all_path),
            "coco_csv": str(coco_path),
        },
        "subset_definitions": {
            "clean": "No detection failure.",
            "valid_rehab": "No detection failure, no >100% outlier, no MoCap annotation error.",
            "valid_coco": "No detection failure, no >100% outlier.",
            "coach_extreme_case": "5 identified coach videos on c17: PM_010, PM_011, PM_108, PM_119, PM_121.",
            "segmentation_multi_person": "Rows with extra_person_severity >= 2.",
        },
        "rehab24_main_models": {
            "metadata": rehab_metadata,
            "clean_n_rows": int(len(rehab_clean)),
            "valid_n_rows": int(len(rehab_valid)),
            "n_video_exercise_pairs": n_video_ids,
            "n_segmentation_person_clusters": n_person_clusters,
            "summary_valid": _records(summarize_model_central_tendency(rehab_main_valid)),
            "summary_coach_all": _records(summarize_model_central_tendency(rehab_coach_all)),
            "summary_coach_valid": _records(summarize_model_central_tendency(rehab_coach_valid)),
            "summary_non_coach_all": _records(summarize_model_central_tendency(rehab_non_coach_all)),
            "summary_segmentation_multi_valid": _records(
                summarize_model_central_tendency(rehab_seg_multi_valid)
            ),
        },
        "rehab24_all_variants": {
            "metadata": rehab_all_metadata,
            "valid_n_rows": int(len(rehab_all_valid)),
            "summary_valid": _records(summarize_model_central_tendency(rehab_all_valid)),
        },
        "coco": {
            "metadata": coco_metadata,
            "valid_n_rows": int(len(coco_valid)),
            "summary_valid": _records(summarize_model_central_tendency(coco_valid)),
        },
        "warnings": warnings,
        "notes": [
            "This report is the provenance layer for thesis-facing numbers during the recovery process.",
            "Legacy numbers under analysis/results or older markdown docs are historical context unless regenerated from evaluation_v2.",
        ],
    }


def write_report(report: dict) -> None:
    out_json = RESULTS_ROOT / "canonical_provenance.json"
    out_md = RESULTS_ROOT / "canonical_provenance.md"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# Canonical Provenance Report",
        "",
        "This file freezes the current thesis evidence base during the recovery process.",
        "",
        "## Authoritative Files",
        "",
        f"- `evaluation_v2/results/rehab24/frame_results.csv`",
        f"- `evaluation_v2/results/rehab24_all/frame_results_all.csv`",
        f"- `evaluation_v2/results/coco/coco_results.csv`",
        "",
        "## Subset Definitions",
        "",
    ]
    for key, value in report["subset_definitions"].items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(
        [
            "",
            "## Rehab24 Main Models (Valid Subset)",
            "",
            "| Model | Mean NMPJPE | Median | Std | Frames |",
            "|-------|-------------|--------|-----|--------|",
        ]
    )
    for row in report["rehab24_main_models"]["summary_valid"]:
        lines.append(
            f"| {row['model']} | {row['mean_nmpjpe']:.2f}% | {row['median_nmpjpe']:.2f}% | "
            f"{row['std_nmpjpe']:.2f}% | {int(row['n_frames'])} |"
        )

    lines.extend(
        [
            "",
            "## Rehab24 All Variants (Valid Subset)",
            "",
            "| Model | Mean NMPJPE | Median | Frames |",
            "|-------|-------------|--------|--------|",
        ]
    )
    for row in report["rehab24_all_variants"]["summary_valid"]:
        lines.append(
            f"| {row['model']} | {row['mean_nmpjpe']:.2f}% | {row['median_nmpjpe']:.2f}% | "
            f"{int(row['n_frames'])} |"
        )

    lines.extend(
        [
            "",
            "## COCO (Valid Subset)",
            "",
            "| Model | Mean NMPJPE | Median | Frames |",
            "|-------|-------------|--------|--------|",
        ]
    )
    for row in report["coco"]["summary_valid"]:
        lines.append(
            f"| {row['model']} | {row['mean_nmpjpe']:.2f}% | {row['median_nmpjpe']:.2f}% | "
            f"{int(row['n_frames'])} |"
        )

    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


def main() -> None:
    report = build_report()
    write_report(report)


if __name__ == "__main__":
    main()
