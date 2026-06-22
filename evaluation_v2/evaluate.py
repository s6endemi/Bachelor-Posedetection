"""
Evaluation Pipeline v2 — Core Evaluation

Single codepath: .npz predictions + GT → results CSV.

Usage:
    python evaluation_v2/evaluate.py
    python evaluation_v2/evaluate.py --dataset rehab24

Output: evaluation_v2/results/{dataset}/frame_results.csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_v2.config import (
    COMPARABLE_COCO_INDICES,
    COMPARABLE_JOINT_NAMES,
    MODEL_DISPLAY_NAMES,
    DatasetConfig,
    EvalConfig,
    get_rehab24_config,
)


# =============================================================================
# Core Metric Computation
# =============================================================================

def compute_torso_length(gt_frame_12: np.ndarray) -> float:
    """
    Torso length = distance from shoulder-center to hip-center.

    Args:
        gt_frame_12: GT keypoints for one frame, shape (12, 2).
                     Indices 0,1 = shoulders; 6,7 = hips.
    Returns:
        Torso length in pixels. Returns NaN if zero (degenerate).
    """
    shoulder_center = (gt_frame_12[0] + gt_frame_12[1]) / 2
    hip_center = (gt_frame_12[6] + gt_frame_12[7]) / 2
    length = np.linalg.norm(shoulder_center - hip_center)
    return length if length > 1.0 else np.nan


def compute_rotation_angle(
    gt_3d_frame: np.ndarray,
    camera: str,
    left_shoulder_idx: int,
    right_shoulder_idx: int,
    c17_offset: float,
) -> float:
    """
    Camera-relative body rotation from 3D shoulder positions.

    Returns angle in [0, 90] degrees where:
        0° = frontal to camera
        90° = lateral (side view)
    """
    left = gt_3d_frame[left_shoulder_idx, :3]
    right = gt_3d_frame[right_shoulder_idx, :3]
    dx = right[0] - left[0]
    dz = right[2] - left[2]
    mocap_angle = np.degrees(np.arctan2(abs(dz), abs(dx)))
    c17_relative = abs(mocap_angle - c17_offset)
    if camera == "c17":
        return np.clip(c17_relative, 0, 90)
    else:
        return np.clip(90.0 - c17_relative, 0, 90)


def evaluate_frame(
    pred_17: np.ndarray,
    gt_2d_26: np.ndarray,
    dataset: DatasetConfig,
    eval_cfg: EvalConfig,
) -> dict:
    """
    Evaluate a single frame.

    Args:
        pred_17: Predictions, shape (17, 3) — [x, y, confidence] in COCO order.
        gt_2d_26: Ground truth, shape (26, 2) — [x, y] in GT joint order.
        dataset: Dataset config for keypoint mapping.
        eval_cfg: Evaluation parameters.

    Returns:
        Dict with: nmpjpe, n_valid_joints, per_joint_errors, per_joint_valid,
                    torso_length, is_detection_failure.
    """
    # Extract 12 comparable joints from GT
    gt_12 = np.zeros((12, 2))
    for i, coco_idx in enumerate(COMPARABLE_COCO_INDICES):
        gt_idx = {v: k for k, v in dataset.gt_to_coco.items()}[coco_idx]
        gt_12[i] = gt_2d_26[gt_idx]

    # Extract 12 comparable joints from predictions
    pred_12_xy = pred_17[COMPARABLE_COCO_INDICES, :2]
    pred_12_conf = pred_17[COMPARABLE_COCO_INDICES, 2]

    # Torso length for normalization
    torso_length = compute_torso_length(gt_12)
    if np.isnan(torso_length):
        return {
            "nmpjpe": np.nan,
            "n_valid_joints": 0,
            "per_joint_errors": [np.nan] * 12,
            "per_joint_valid": [False] * 12,
            "torso_length": np.nan,
            "is_detection_failure": True,
        }

    # Per-joint Euclidean error, normalized by torso
    raw_errors = np.linalg.norm(pred_12_xy - gt_12, axis=1)
    normalized_errors = raw_errors / torso_length * 100  # percent of torso

    # Confidence filter
    valid_mask = pred_12_conf >= eval_cfg.joint_confidence_threshold

    # Check for detection failure: all joints at (0,0) or all below threshold
    if not np.any(valid_mask) or np.all(pred_12_xy == 0):
        return {
            "nmpjpe": np.nan,
            "n_valid_joints": 0,
            "per_joint_errors": [np.nan] * 12,
            "per_joint_valid": valid_mask.tolist(),
            "torso_length": torso_length,
            "is_detection_failure": True,
        }

    # NMPJPE = mean of valid joint errors
    filtered_errors = np.where(valid_mask, normalized_errors, np.nan)
    nmpjpe = float(np.nanmean(filtered_errors))

    return {
        "nmpjpe": nmpjpe,
        "n_valid_joints": int(np.sum(valid_mask)),
        "per_joint_errors": [
            float(e) if v else np.nan
            for e, v in zip(normalized_errors, valid_mask)
        ],
        "per_joint_valid": valid_mask.tolist(),
        "torso_length": torso_length,
        "is_detection_failure": False,
    }


# =============================================================================
# Segmentation Metadata (Multi-Person Flagging)
# =============================================================================

def load_segmentation(dataset: DatasetConfig, eval_cfg: EvalConfig) -> dict:
    """
    Load Segmentation.csv and build a lookup for multi-person frame ranges.

    Returns:
        Dict mapping (video_id, exercise, camera) -> list of
        {first_frame, last_frame, severity, orientation, mocap_error}
    """
    if dataset.segmentation_csv is None or not dataset.segmentation_csv.exists():
        return {}

    df = pd.read_csv(dataset.segmentation_csv, delimiter=dataset.seg_delimiter)
    lookup = {}

    for _, row in df.iterrows():
        video_id = row[dataset.seg_video_col]
        exercise = f"Ex{int(row[dataset.seg_exercise_col])}"

        for camera in ["c17", "c18"]:
            col = (
                dataset.seg_extra_person_c17_col
                if camera == "c17"
                else dataset.seg_extra_person_c18_col
            )
            severity = int(row[col])
            key = (video_id, exercise, camera)
            entry = {
                "first_frame": int(row[dataset.seg_first_frame_col]),
                "last_frame": int(row[dataset.seg_last_frame_col]),
                "extra_person_severity": severity,
                "orientation": row.get(dataset.seg_orientation_col, ""),
                "mocap_erroneous": int(row.get(dataset.seg_mocap_error_col, 0)),
            }
            lookup.setdefault(key, []).append(entry)

    return lookup


def get_frame_metadata(
    seg_lookup: dict,
    video_id: str,
    exercise: str,
    camera: str,
    video_frame_idx: int,
    eval_cfg: EvalConfig,
) -> dict:
    """
    For a specific video frame, return multi-person severity and metadata.
    """
    key = (video_id, exercise, camera)
    segments = seg_lookup.get(key, [])

    best_match = {"extra_person_severity": 0, "orientation": "", "mocap_erroneous": 0}
    for seg in segments:
        if seg["first_frame"] <= video_frame_idx <= seg["last_frame"]:
            best_match = seg
            break

    is_multi_person = (
        best_match["extra_person_severity"] >= eval_cfg.multi_person_severity_threshold
    )
    return {
        "extra_person_severity": best_match["extra_person_severity"],
        "is_multi_person": is_multi_person,
        "orientation": best_match["orientation"],
        "mocap_erroneous": best_match["mocap_erroneous"],
    }


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def discover_videos(dataset: DatasetConfig) -> list[dict]:
    """Find all .npz prediction files and match with GT."""
    videos = []
    for exercise_dir in sorted(dataset.predictions_dir.iterdir()):
        if not exercise_dir.is_dir():
            continue
        exercise = exercise_dir.name  # e.g., "Ex1"

        for npz_file in sorted(exercise_dir.glob("*.npz")):
            # Parse filename: PM_000-c17.npz
            stem = npz_file.stem  # PM_000-c17
            parts = stem.rsplit("-", 1)
            if len(parts) != 2:
                continue
            video_id, camera = parts  # PM_000, c17

            # Find matching GT files
            gt_2d_file = dataset.gt_2d_dir / exercise / f"{stem}-30fps.npy"
            gt_3d_file = dataset.gt_3d_dir / exercise / f"{video_id}-30fps.npy"

            if not gt_2d_file.exists() or not gt_3d_file.exists():
                print(f"  WARNING: GT missing for {stem}, skipping")
                continue

            videos.append({
                "npz_path": npz_file,
                "gt_2d_path": gt_2d_file,
                "gt_3d_path": gt_3d_file,
                "exercise": exercise,
                "video_id": video_id,
                "camera": camera,
            })

    return videos


def evaluate_video(
    video: dict,
    dataset: DatasetConfig,
    eval_cfg: EvalConfig,
    seg_lookup: dict,
) -> list[dict]:
    """Evaluate all frames of one video for all models."""
    npz_data = np.load(video["npz_path"])
    gt_2d = np.load(video["gt_2d_path"])
    gt_3d = np.load(video["gt_3d_path"])

    n_pred_frames = int(npz_data["num_frames"])
    rotation_angles = npz_data["rotation_angles"]

    rows = []

    for frame_i in range(n_pred_frames):
        # Frame alignment: prediction[i] corresponds to video_frame[i * frame_step]
        video_frame_idx = frame_i * eval_cfg.frame_step

        if video_frame_idx >= gt_2d.shape[0]:
            break

        gt_2d_frame = gt_2d[video_frame_idx]
        rotation = float(rotation_angles[frame_i])

        # Multi-person metadata from segmentation
        meta = get_frame_metadata(
            seg_lookup,
            video["video_id"],
            video["exercise"],
            video["camera"],
            video_frame_idx,
            eval_cfg,
        )

        # Evaluate each model
        for model_key in dataset.model_keys:
            if model_key not in npz_data:
                continue

            pred_frame = npz_data[model_key][frame_i]
            result = evaluate_frame(pred_frame, gt_2d_frame, dataset, eval_cfg)

            row = {
                "video_id": video["video_id"],
                "exercise": video["exercise"],
                "camera": video["camera"],
                "frame_idx": frame_i,
                "video_frame_idx": video_frame_idx,
                "model": MODEL_DISPLAY_NAMES.get(model_key, model_key),
                "rotation_angle": rotation,
                "rotation_bin": int(rotation // eval_cfg.rotation_bin_size)
                * eval_cfg.rotation_bin_size,
                "nmpjpe": result["nmpjpe"],
                "n_valid_joints": result["n_valid_joints"],
                "torso_length": result["torso_length"],
                "is_detection_failure": result["is_detection_failure"],
                "is_outlier": (
                    result["nmpjpe"] > eval_cfg.outlier_nmpjpe_threshold
                    if not np.isnan(result["nmpjpe"])
                    else False
                ),
                "extra_person_severity": meta["extra_person_severity"],
                "is_multi_person": meta["is_multi_person"],
                "orientation": meta["orientation"],
                "mocap_erroneous": meta["mocap_erroneous"],
            }

            # Per-joint errors
            for j, name in enumerate(COMPARABLE_JOINT_NAMES):
                row[f"error_{name}"] = result["per_joint_errors"][j]
                row[f"valid_{name}"] = result["per_joint_valid"][j]

            rows.append(row)

    return rows


def compute_jitter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy temporal-stability proxy based on frame-to-frame NMPJPE change.

    Important:
    - This function does NOT compute true keypoint displacement jitter.
    - It is retained for backward compatibility with older result files.
    - Canonical thesis jitter should be recomputed from stored prediction arrays
      via `evaluation_v2/recompute_true_jitter.py`.
    """
    jitter_rows = []

    for (model, video_id, exercise, camera), group in df.groupby(
        ["model", "video_id", "exercise", "camera"]
    ):
        group = group.sort_values("frame_idx")
        if len(group) < 2:
            continue

        frame_indices = group["frame_idx"].values
        nmpjpe_values = group["nmpjpe"].values
        torso_values = group["torso_length"].values

        for i in range(1, len(group)):
            if frame_indices[i] != frame_indices[i - 1] + 1:
                continue  # skip non-consecutive frames

            prev_nmpjpe = nmpjpe_values[i - 1]
            curr_nmpjpe = nmpjpe_values[i]

            if np.isnan(prev_nmpjpe) or np.isnan(curr_nmpjpe):
                continue

            # Legacy proxy: absolute change in frame-level NMPJPE
            nmpjpe_jitter = abs(curr_nmpjpe - prev_nmpjpe)

            jitter_rows.append({
                "model": model,
                "video_id": video_id,
                "exercise": exercise,
                "camera": camera,
                "frame_idx": frame_indices[i],
                "nmpjpe_jitter": nmpjpe_jitter,
                "rotation_angle": group.iloc[i]["rotation_angle"],
                "is_multi_person": group.iloc[i]["is_multi_person"],
            })

    return pd.DataFrame(jitter_rows) if jitter_rows else pd.DataFrame()


# =============================================================================
# Entry Point
# =============================================================================

def run_evaluation(dataset: DatasetConfig, eval_cfg: EvalConfig) -> Path:
    """Run full evaluation. Returns path to output CSV."""
    print(f"=== Evaluation v2: {dataset.name} ===")
    print(f"  Predictions: {dataset.predictions_dir}")
    print(f"  GT 2D:       {dataset.gt_2d_dir}")
    print(f"  Output:      {dataset.output_dir}")
    print(f"  Confidence:  {eval_cfg.joint_confidence_threshold}")
    print(f"  Frame step:  {eval_cfg.frame_step}")
    print()

    # Discover videos
    videos = discover_videos(dataset)
    print(f"Found {len(videos)} video-GT pairs")

    # Load segmentation metadata
    seg_lookup = load_segmentation(dataset, eval_cfg)
    if seg_lookup:
        n_multi = sum(
            1
            for segs in seg_lookup.values()
            for s in segs
            if s["extra_person_severity"] >= eval_cfg.multi_person_severity_threshold
        )
        print(f"Loaded segmentation: {len(seg_lookup)} entries, {n_multi} multi-person segments")
    else:
        print("No segmentation metadata found")
    print()

    # Evaluate all videos
    all_rows = []
    for i, video in enumerate(videos):
        label = f"{video['exercise']}/{video['video_id']}-{video['camera']}"
        print(f"  [{i+1}/{len(videos)}] {label}", end="")
        rows = evaluate_video(video, dataset, eval_cfg, seg_lookup)
        all_rows.extend(rows)
        n_models = len(dataset.model_keys)
        n_frames = len(rows) // n_models if n_models else 0
        print(f" — {n_frames} frames × {n_models} models")

    # Build DataFrame
    df = pd.DataFrame(all_rows)
    print(f"\nTotal rows: {len(df):,}")

    # Compute legacy jitter proxy for backward compatibility.
    print("Computing legacy jitter proxy...")
    jitter_df = compute_jitter(df)

    # Save outputs
    dataset.output_dir.mkdir(parents=True, exist_ok=True)

    frame_csv = dataset.output_dir / "frame_results.csv"
    df.to_csv(frame_csv, index=False)
    print(f"Saved: {frame_csv}")

    if not jitter_df.empty:
        jitter_csv = dataset.output_dir / "jitter_results.csv"
        jitter_df.to_csv(jitter_csv, index=False)
        print(f"Saved: {jitter_csv}")

    # Save metadata
    metadata = {
        "dataset": dataset.name,
        "timestamp": datetime.now().isoformat(),
        "eval_config": {
            "joint_confidence_threshold": eval_cfg.joint_confidence_threshold,
            "outlier_nmpjpe_threshold": eval_cfg.outlier_nmpjpe_threshold,
            "frame_step": eval_cfg.frame_step,
            "c17_frontal_offset": eval_cfg.c17_frontal_offset,
            "rotation_bin_size": eval_cfg.rotation_bin_size,
            "multi_person_severity_threshold": eval_cfg.multi_person_severity_threshold,
        },
        "n_videos": len(videos),
        "n_frames_total": len(df),
        "n_models": len(dataset.model_keys),
        "models": list(MODEL_DISPLAY_NAMES.values()),
    }
    meta_path = dataset.output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {meta_path}")

    # Quick summary
    print("\n=== Quick Summary ===")
    valid = df[~df["is_detection_failure"] & ~df["is_outlier"]]
    for model in valid["model"].unique():
        m = valid[valid["model"] == model]
        print(
            f"  {model:12s}: NMPJPE {m['nmpjpe'].mean():.1f}% "
            f"(median {m['nmpjpe'].median():.1f}%, n={len(m):,})"
        )

    # Multi-person breakdown
    multi = valid[valid["is_multi_person"]]
    clean = valid[~valid["is_multi_person"]]
    if len(multi) > 0:
        print(f"\n  Multi-person frames: {len(multi):,} ({len(multi)/len(valid)*100:.1f}%)")
        for model in valid["model"].unique():
            c = clean[clean["model"] == model]["nmpjpe"].mean()
            mp = multi[multi["model"] == model]["nmpjpe"].mean()
            print(f"    {model:12s}: clean {c:.1f}% -> multi-person {mp:.1f}% (+{(mp/c-1)*100:.0f}%)")

    return frame_csv


def main():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline v2")
    parser.add_argument(
        "--dataset",
        default="rehab24",
        help="Dataset to evaluate (default: rehab24)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override joint confidence threshold",
    )
    args = parser.parse_args()

    eval_cfg = EvalConfig()
    if args.confidence is not None:
        eval_cfg.joint_confidence_threshold = args.confidence

    if args.dataset == "rehab24":
        dataset = get_rehab24_config(PROJECT_ROOT)
    else:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    run_evaluation(dataset, eval_cfg)


if __name__ == "__main__":
    main()
