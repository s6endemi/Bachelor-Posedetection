"""
Full Evaluation: All 9 model variants on REHAB24-6.

Reads baseline .npz (3 models) + variant .npz (6 models) + GT
-> produces one unified frame_results_all.csv with 41 columns per frame.

Usage:
    python evaluation_v2/evaluate_all.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_v2.config import (
    COMPARABLE_COCO_INDICES,
    COMPARABLE_JOINT_NAMES,
    EvalConfig,
    get_rehab24_config,
)
from evaluation_v2.evaluate import (
    evaluate_frame,
    load_segmentation,
    get_frame_metadata,
    compute_jitter,
)


# Model definitions: (display_name, predictions_dir, npz_key)
MODELS = [
    # Baselines (in data/predictions/)
    ("MediaPipe_Full", "predictions", "pred_MediaPipe_full"),
    ("MoveNet_MultiPose", "predictions", "pred_MoveNet_multipose"),
    ("YOLOv8_Nano", "predictions", "pred_YOLOv8-Pose_n"),
    # Variants (in data/predictions_variants/{name}/)
    ("MediaPipe_Lite", "predictions_variants/MediaPipe_Lite", "pred_MediaPipe_Lite"),
    ("MediaPipe_Heavy", "predictions_variants/MediaPipe_Heavy", "pred_MediaPipe_Heavy"),
    ("MoveNet_SP_Lightning", "predictions_variants/MoveNet_SP_Lightning", "pred_MoveNet_SP_Lightning"),
    ("MoveNet_SP_Thunder", "predictions_variants/MoveNet_SP_Thunder", "pred_MoveNet_SP_Thunder"),
    ("YOLOv8_Small", "predictions_variants/YOLOv8s", "pred_YOLOv8s"),
    ("YOLOv8_Medium", "predictions_variants/YOLOv8m", "pred_YOLOv8m"),
]


def discover_videos(data_dir: Path) -> list[dict]:
    """Find all videos using baseline predictions as reference."""
    videos = []
    pred_dir = data_dir / "predictions"
    for ex_dir in sorted(pred_dir.iterdir()):
        if not ex_dir.is_dir():
            continue
        exercise = ex_dir.name
        for npz_file in sorted(ex_dir.glob("*.npz")):
            stem = npz_file.stem
            parts = stem.rsplit("-", 1)
            if len(parts) != 2:
                continue
            video_id, camera = parts
            gt_2d_file = data_dir / "gt_2d" / exercise / f"{stem}-30fps.npy"
            if not gt_2d_file.exists():
                continue
            videos.append({
                "exercise": exercise,
                "video_id": video_id,
                "camera": camera,
                "stem": stem,
                "gt_2d_path": gt_2d_file,
            })
    return videos


def evaluate_all():
    data_dir = PROJECT_ROOT / "data"
    dataset = get_rehab24_config(PROJECT_ROOT)
    eval_cfg = EvalConfig()
    output_dir = PROJECT_ROOT / "evaluation_v2" / "results" / "rehab24_all"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Full Evaluation: 9 Variants on REHAB24-6 ===")
    print(f"  Confidence: {eval_cfg.joint_confidence_threshold}")
    print(f"  Output: {output_dir}")
    print()

    # Load segmentation
    seg_lookup = load_segmentation(dataset, eval_cfg)
    print(f"Segmentation: {len(seg_lookup)} entries loaded")

    # Discover videos
    videos = discover_videos(data_dir)
    print(f"Videos: {len(videos)}")
    print(f"Models: {len(MODELS)}")
    print()

    all_rows = []
    t_start = time.time()

    for vi, video in enumerate(videos):
        gt_2d = np.load(video["gt_2d_path"])

        # Load baseline .npz (has rotation angles)
        baseline_npz = np.load(data_dir / "predictions" / video["exercise"] / f"{video['stem']}.npz")
        rotation_angles = baseline_npz["rotation_angles"]
        n_frames = int(baseline_npz["num_frames"])

        for model_name, pred_dir, pred_key in MODELS:
            # Load predictions
            npz_path = data_dir / pred_dir / video["exercise"] / f"{video['stem']}.npz"
            if not npz_path.exists():
                continue

            npz_data = np.load(npz_path)
            if pred_key not in npz_data:
                continue

            predictions = npz_data[pred_key]

            for frame_i in range(min(n_frames, len(predictions))):
                video_frame_idx = frame_i * eval_cfg.frame_step
                if video_frame_idx >= gt_2d.shape[0]:
                    break

                gt_2d_frame = gt_2d[video_frame_idx]
                rotation = float(rotation_angles[frame_i]) if frame_i < len(rotation_angles) else 0

                # Multi-person metadata
                meta = get_frame_metadata(
                    seg_lookup, video["video_id"], video["exercise"],
                    video["camera"], video_frame_idx, eval_cfg,
                )

                # Evaluate
                pred_frame = predictions[frame_i]
                result = evaluate_frame(pred_frame, gt_2d_frame, dataset, eval_cfg)

                row = {
                    "video_id": video["video_id"],
                    "exercise": video["exercise"],
                    "camera": video["camera"],
                    "frame_idx": frame_i,
                    "video_frame_idx": video_frame_idx,
                    "model": model_name,
                    "rotation_angle": rotation,
                    "rotation_bin": int(rotation // eval_cfg.rotation_bin_size) * eval_cfg.rotation_bin_size,
                    "nmpjpe": result["nmpjpe"],
                    "n_valid_joints": result["n_valid_joints"],
                    "torso_length": result["torso_length"],
                    "is_detection_failure": result["is_detection_failure"],
                    "is_outlier": (
                        result["nmpjpe"] > eval_cfg.outlier_nmpjpe_threshold
                        if not np.isnan(result["nmpjpe"]) else False
                    ),
                    "extra_person_severity": meta["extra_person_severity"],
                    "is_multi_person": meta["is_multi_person"],
                    "orientation": meta["orientation"],
                    "mocap_erroneous": meta["mocap_erroneous"],
                }
                for j, name in enumerate(COMPARABLE_JOINT_NAMES):
                    row[f"error_{name}"] = result["per_joint_errors"][j]
                    row[f"valid_{name}"] = result["per_joint_valid"][j]

                all_rows.append(row)

        if (vi + 1) % 10 == 0 or (vi + 1) == len(videos):
            elapsed = time.time() - t_start
            vps = (vi + 1) / elapsed
            eta = (len(videos) - vi - 1) / vps / 60 if vps > 0 else 0
            print(f"  [{vi+1}/{len(videos)}] {video['stem']} | {len(all_rows):,} rows | ETA {eta:.0f}min")

    # Build DataFrame
    df = pd.DataFrame(all_rows)
    print(f"\nTotal rows: {len(df):,}")

    # Save frame results
    csv_path = output_dir / "frame_results_all.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Compute legacy jitter proxy. Canonical true jitter is recomputed separately.
    print("Computing legacy jitter proxy...")
    jitter_df = compute_jitter(df)
    if not jitter_df.empty:
        jitter_path = output_dir / "jitter_results_all.csv"
        jitter_df.to_csv(jitter_path, index=False)
        print(f"Saved: {jitter_path}")

    # Save metadata
    metadata = {
        "dataset": "REHAB24-6",
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
        "models": [m[0] for m in MODELS],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print(f"\n=== Summary (valid frames, no outliers, no mocap errors) ===")
    valid = df[~df["is_detection_failure"] & ~df["is_outlier"] & (df["mocap_erroneous"] == 0)]

    results = []
    for model in valid["model"].unique():
        m = valid[valid["model"] == model]
        results.append((model, m["nmpjpe"].mean(), m["nmpjpe"].median(), len(m)))
    results.sort(key=lambda x: x[1])

    print(f"{'Rang':>4}  {'Modell':25s} {'NMPJPE':>8s} {'Median':>8s} {'Frames':>10s}")
    print("-" * 60)
    for i, (model, mean, med, n) in enumerate(results):
        print(f"{i+1:4d}. {model:25s} {mean:7.1f}% {med:7.1f}% {n:>10,}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    evaluate_all()
