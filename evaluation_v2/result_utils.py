"""
Shared helpers for canonical thesis result subsets.

The main goal is to keep all analysis scripts on the same filtering logic:
- clean: no detection failure
- valid: no detection failure, no >100% outlier, no MoCap error
- coach extreme-case subset: the 5 identified coach videos on c17
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_ROOT = PROJECT_ROOT / "evaluation_v2" / "results"

MAIN_MODELS = ["MediaPipe", "MoveNet", "YOLOv8"]
MAIN_VARIANT_MODELS = ["MediaPipe_Full", "MoveNet_MultiPose", "YOLOv8_Nano"]
ALL_VARIANT_MODELS = [
    "MediaPipe_Full",
    "MoveNet_MultiPose",
    "YOLOv8_Nano",
    "MediaPipe_Lite",
    "MediaPipe_Heavy",
    "MoveNet_SP_Lightning",
    "MoveNet_SP_Thunder",
    "YOLOv8_Small",
    "YOLOv8_Medium",
]
COCO_MODELS = [
    "MediaPipe",
    "MediaPipe_Heavy",
    "MediaPipe_Lite",
    "MoveNet",
    "MoveNet_SP_Lightning",
    "MoveNet_SP_Thunder",
    "YOLOv8",
    "YOLOv8m",
    "YOLOv8s",
]
COACH_VIDEO_IDS = {"PM_010", "PM_011", "PM_108", "PM_119", "PM_121"}


@lru_cache(maxsize=1)
def get_segmentation_person_mapping() -> pd.DataFrame:
    """
    Map REHAB24-6 video/exercise pairs to segmentation person IDs.

    The repository's result files use `video_id` as an exercise-specific sample ID.
    For clustering, the best explicit person grouping currently available in-repo is
    `person_id` from `Segmentation (1).csv`.
    """
    seg_path = PROJECT_ROOT / "data" / "Segmentation (1).csv"
    if not seg_path.exists():
        return pd.DataFrame(columns=["video_id", "exercise", "patient_id"])

    seg = pd.read_csv(seg_path, delimiter=";")
    mapping = seg[["video_id", "exercise_id", "person_id"]].drop_duplicates().copy()
    mapping["exercise"] = "Ex" + mapping["exercise_id"].astype(str)
    mapping["patient_id"] = mapping["person_id"].astype(int)
    return mapping[["video_id", "exercise", "patient_id"]]


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load a result CSV and add shared helper columns."""
    df = pd.read_csv(csv_path)
    if "video_id" in df.columns and "exercise" in df.columns:
        person_map = get_segmentation_person_mapping()
        df = df.merge(person_map, on=["video_id", "exercise"], how="left")
        if "patient_id" not in df.columns:
            df["patient_id"] = pd.NA
        df["patient_id"] = df["patient_id"].fillna(df["video_id"])
        if "camera" in df.columns:
            df["is_coach_video"] = df["video_id"].isin(COACH_VIDEO_IDS) & (df["camera"] == "c17")
        else:
            df["is_coach_video"] = False
    else:
        df["patient_id"] = None
        df["is_coach_video"] = False
    return df


def filter_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rows without full detection failure."""
    if "is_detection_failure" not in df.columns:
        return df.copy()
    return df[~df["is_detection_failure"]].copy()


def filter_valid(df: pd.DataFrame, require_mocap_clean: bool = True) -> pd.DataFrame:
    """
    Canonical subset used for central tendency claims.

    Default for REHAB:
    - no detection failure
    - no >100% outlier
    - no MoCap annotation error
    """
    out = filter_clean(df)
    if "is_outlier" in out.columns:
        out = out[~out["is_outlier"]]
    if require_mocap_clean and "mocap_erroneous" in out.columns:
        out = out[out["mocap_erroneous"] == 0]
    return out.copy()


def filter_main_models(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the 3 main models, regardless of REHAB/COCO naming."""
    model_names = set(MAIN_MODELS) | set(MAIN_VARIANT_MODELS)
    return df[df["model"].isin(model_names)].copy()


def filter_coach_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the 5 identified coach videos on c17."""
    if "is_coach_video" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["is_coach_video"]].copy()


def filter_non_coach_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude the 5 identified coach videos on c17."""
    if "is_coach_video" not in df.columns:
        return df.copy()
    return df[~df["is_coach_video"]].copy()


def filter_segmentation_multi_person(
    df: pd.DataFrame,
    severity_threshold: int = 2,
) -> pd.DataFrame:
    """Rows with segmentation-defined multi-person severity above threshold."""
    if "extra_person_severity" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["extra_person_severity"] >= severity_threshold].copy()


def summarize_model_central_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """Simple model summary for NMPJPE mean/median/std and frame counts."""
    summary = (
        df.groupby("model", dropna=False)["nmpjpe"]
        .agg(mean_nmpjpe="mean", median_nmpjpe="median", std_nmpjpe="std", n_frames="size")
        .reset_index()
        .sort_values("mean_nmpjpe")
    )
    return summary
