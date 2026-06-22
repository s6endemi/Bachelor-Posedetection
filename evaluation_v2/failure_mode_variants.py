"""
Within-family consistency check for the failure-mode taxonomy.

Runs the same classification as failure_mode_analysis.py on all nine model
variants (rehab24_all) to test whether the dominant failure mode of each
architectural paradigm is consistent across family members. A consistent
pattern (e.g., all MediaPipe variants keypoint-displacement-dominant) reduces
the risk that the main-model result reflects a single-model artifact.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_v2.failure_mode_analysis import (
    CATEGORY_ORDER,
    COLLAPSE_THRESHOLD_DEFAULT,
    classify,
    per_model_counts,
)
from evaluation_v2.result_utils import ALL_VARIANT_MODELS, load_results

FAMILY_MAP = {
    "MediaPipe_Full":       "MediaPipe",
    "MediaPipe_Lite":       "MediaPipe",
    "MediaPipe_Heavy":      "MediaPipe",
    "MoveNet_MultiPose":    "MoveNet",
    "MoveNet_SP_Lightning": "MoveNet",
    "MoveNet_SP_Thunder":   "MoveNet",
    "YOLOv8_Nano":          "YOLOv8",
    "YOLOv8_Small":         "YOLOv8",
    "YOLOv8_Medium":        "YOLOv8",
}


def main() -> None:
    results_path = PROJECT_ROOT / "evaluation_v2" / "results" / "rehab24_all" / "frame_results_all.csv"
    df = load_results(results_path)
    df = df[df["model"].isin(ALL_VARIANT_MODELS)].copy()

    classified = classify(df, COLLAPSE_THRESHOLD_DEFAULT)
    counts = per_model_counts(classified)

    print(f"Collapse threshold = {COLLAPSE_THRESHOLD_DEFAULT}\n")
    print("| Variant | Family | Failures | MissDet% | ConfColl% | MultiPers% | KpDispl% | Dominant |")
    print("|---------|--------|---------:|---------:|----------:|-----------:|---------:|----------|")
    family_order = ["MediaPipe", "MoveNet", "YOLOv8"]
    for family in family_order:
        fam_members = [m for m in ALL_VARIANT_MODELS if FAMILY_MAP[m] == family]
        for v in fam_members:
            if v not in counts.index:
                continue
            row = counts.loc[v]
            nf = int(row["n_failure"])
            pcts = {cat: row[cat] / nf * 100 if nf else 0.0 for cat in CATEGORY_ORDER}
            dominant = max(CATEGORY_ORDER, key=lambda c: pcts[c])
            print(
                f"| {v} | {family} | {nf:,} | "
                f"{pcts['Missing-Detection']:5.1f} | "
                f"{pcts['Confidence-Collapse']:5.1f} | "
                f"{pcts['Multi-Person-Confusion']:5.1f} | "
                f"{pcts['Keypoint-Displacement']:5.1f} | "
                f"{dominant} |"
            )
        print("|---|---|---|---|---|---|---|---|")

    # Save to CSV too
    out_path = PROJECT_ROOT / "evaluation_v2" / "results" / "failure_mode_variants.csv"
    records = []
    for v in ALL_VARIANT_MODELS:
        if v not in counts.index:
            continue
        row = counts.loc[v]
        nf = int(row["n_failure"])
        records.append({
            "variant": v,
            "family": FAMILY_MAP[v],
            "n_failure": nf,
            **{f"{cat}_pct": row[cat] / nf * 100 if nf else 0.0 for cat in CATEGORY_ORDER},
            **{cat: int(row[cat]) for cat in CATEGORY_ORDER},
            "dominant": max(CATEGORY_ORDER, key=lambda c: row[c] / nf if nf else 0),
        })
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
