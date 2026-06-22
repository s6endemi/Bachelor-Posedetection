"""
Repair stale COCO metadata from the existing merged result CSV.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_v2.result_utils import filter_valid, load_results


def main() -> None:
    output_dir = PROJECT_ROOT / "evaluation_v2" / "results" / "coco"
    csv_path = output_dir / "coco_results.csv"
    metadata_path = output_dir / "metadata.json"

    df = load_results(csv_path)
    valid = filter_valid(df, require_mocap_clean=False)

    metadata = {
        "dataset": "COCO val2017",
        "timestamp": datetime.now().isoformat(),
        "source": "repaired from existing coco_results.csv",
        "models": sorted(df["model"].dropna().unique().tolist()),
        "confidence": 0.5,
        "max_images": None,
        "total_rows": int(len(df)),
        "valid_rows": int(len(valid)),
        "n_images": int(df["image_id"].nunique()),
        "variants_included": any("_" in model or model.endswith(("s", "m")) for model in df["model"].unique()),
        "single_person_rows": int((df["is_single_person"] == True).sum()),
        "multi_person_rows": int((df["is_single_person"] == False).sum()),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {metadata_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
