"""
Recompute true keypoint-displacement jitter from stored prediction .npz files.

No model rerun is required. The script reads the existing prediction arrays and
normalizes frame-to-frame joint displacement by the average GT torso length of
the two consecutive frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_v2.config import COMPARABLE_COCO_INDICES, COMPARABLE_JOINT_NAMES, EvalConfig, get_rehab24_config
from evaluation_v2.evaluate import compute_torso_length, get_frame_metadata, load_segmentation


BASELINE_MODELS = [
    ("MediaPipe", "predictions", "pred_MediaPipe_full"),
    ("MoveNet", "predictions", "pred_MoveNet_multipose"),
    ("YOLOv8", "predictions", "pred_YOLOv8-Pose_n"),
]

ALL_VARIANT_MODELS = [
    ("MediaPipe_Full", "predictions", "pred_MediaPipe_full"),
    ("MoveNet_MultiPose", "predictions", "pred_MoveNet_multipose"),
    ("YOLOv8_Nano", "predictions", "pred_YOLOv8-Pose_n"),
    ("MediaPipe_Lite", "predictions_variants/MediaPipe_Lite", "pred_MediaPipe_Lite"),
    ("MediaPipe_Heavy", "predictions_variants/MediaPipe_Heavy", "pred_MediaPipe_Heavy"),
    ("MoveNet_SP_Lightning", "predictions_variants/MoveNet_SP_Lightning", "pred_MoveNet_SP_Lightning"),
    ("MoveNet_SP_Thunder", "predictions_variants/MoveNet_SP_Thunder", "pred_MoveNet_SP_Thunder"),
    ("YOLOv8_Small", "predictions_variants/YOLOv8s", "pred_YOLOv8s"),
    ("YOLOv8_Medium", "predictions_variants/YOLOv8m", "pred_YOLOv8m"),
]


def discover_videos(data_dir: Path) -> list[dict]:
    videos = []
    pred_dir = data_dir / "predictions"
    for exercise_dir in sorted(pred_dir.iterdir()):
        if not exercise_dir.is_dir():
            continue
        exercise = exercise_dir.name
        for npz_file in sorted(exercise_dir.glob("*.npz")):
            stem = npz_file.stem
            parts = stem.rsplit("-", 1)
            if len(parts) != 2:
                continue
            video_id, camera = parts
            gt_2d_file = data_dir / "gt_2d" / exercise / f"{stem}-30fps.npy"
            if not gt_2d_file.exists():
                continue
            videos.append(
                {
                    "exercise": exercise,
                    "video_id": video_id,
                    "camera": camera,
                    "stem": stem,
                    "gt_2d_path": gt_2d_file,
                }
            )
    return videos


def extract_gt_12(gt_2d_26: np.ndarray, dataset_cfg) -> np.ndarray:
    gt_12 = np.zeros((12, 2), dtype=float)
    reverse_map = {v: k for k, v in dataset_cfg.gt_to_coco.items()}
    for i, coco_idx in enumerate(COMPARABLE_COCO_INDICES):
        gt_12[i] = gt_2d_26[reverse_map[coco_idx]]
    return gt_12


def compute_pair_torso_length(gt_prev_26: np.ndarray, gt_curr_26: np.ndarray, dataset_cfg) -> float:
    prev_torso = compute_torso_length(extract_gt_12(gt_prev_26, dataset_cfg))
    curr_torso = compute_torso_length(extract_gt_12(gt_curr_26, dataset_cfg))
    valid = [value for value in (prev_torso, curr_torso) if not np.isnan(value)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def compute_video_jitter_rows(
    video: dict,
    models: list[tuple[str, str, str]],
    data_dir: Path,
    dataset_cfg,
    eval_cfg: EvalConfig,
    seg_lookup: dict,
) -> list[dict]:
    gt_2d = np.load(video["gt_2d_path"])
    rows = []

    for model_name, pred_dir, pred_key in models:
        npz_path = data_dir / pred_dir / video["exercise"] / f"{video['stem']}.npz"
        if not npz_path.exists():
            continue
        npz_data = np.load(npz_path)
        if pred_key not in npz_data:
            continue

        predictions = npz_data[pred_key]
        rotation_angles = npz_data["rotation_angles"]
        n_frames = min(int(npz_data["num_frames"]), len(predictions), len(rotation_angles))

        for frame_i in range(1, n_frames):
            prev_video_frame_idx = (frame_i - 1) * eval_cfg.frame_step
            curr_video_frame_idx = frame_i * eval_cfg.frame_step
            if curr_video_frame_idx >= gt_2d.shape[0]:
                break

            pred_prev = predictions[frame_i - 1]
            pred_curr = predictions[frame_i]
            pred_prev_xy = pred_prev[COMPARABLE_COCO_INDICES, :2]
            pred_curr_xy = pred_curr[COMPARABLE_COCO_INDICES, :2]
            pred_prev_conf = pred_prev[COMPARABLE_COCO_INDICES, 2]
            pred_curr_conf = pred_curr[COMPARABLE_COCO_INDICES, 2]
            valid_mask = (
                (pred_prev_conf >= eval_cfg.joint_confidence_threshold)
                & (pred_curr_conf >= eval_cfg.joint_confidence_threshold)
            )

            torso_length = compute_pair_torso_length(
                gt_2d[prev_video_frame_idx],
                gt_2d[curr_video_frame_idx],
                dataset_cfg,
            )
            if np.isnan(torso_length) or not np.any(valid_mask):
                continue

            displacements = np.linalg.norm(pred_curr_xy - pred_prev_xy, axis=1)
            normalized = displacements / torso_length * 100
            filtered = np.where(valid_mask, normalized, np.nan)

            meta = get_frame_metadata(
                seg_lookup,
                video["video_id"],
                video["exercise"],
                video["camera"],
                curr_video_frame_idx,
                eval_cfg,
            )

            row = {
                "model": model_name,
                "video_id": video["video_id"],
                "exercise": video["exercise"],
                "camera": video["camera"],
                "frame_idx": frame_i,
                "video_frame_idx": curr_video_frame_idx,
                "rotation_angle": float(rotation_angles[frame_i]),
                "is_multi_person": meta["is_multi_person"],
                "extra_person_severity": meta["extra_person_severity"],
                "keypoint_jitter": float(np.nanmean(filtered)),
                "median_joint_jitter": float(np.nanmedian(filtered)),
                "n_valid_joints": int(np.sum(valid_mask)),
                "torso_length": torso_length,
            }
            for joint_idx, joint_name in enumerate(COMPARABLE_JOINT_NAMES):
                row[f"jitter_{joint_name}"] = float(filtered[joint_idx]) if valid_mask[joint_idx] else np.nan
            rows.append(row)

    return rows


def run(dataset_name: str) -> Path:
    if dataset_name not in {"rehab24", "rehab24_all"}:
        raise ValueError("dataset must be one of: rehab24, rehab24_all")

    models = BASELINE_MODELS if dataset_name == "rehab24" else ALL_VARIANT_MODELS
    out_name = "jitter_results.csv" if dataset_name == "rehab24" else "jitter_results_all.csv"
    out_dir = PROJECT_ROOT / "evaluation_v2" / "results" / dataset_name

    data_dir = PROJECT_ROOT / "data"
    dataset_cfg = get_rehab24_config(PROJECT_ROOT)
    eval_cfg = EvalConfig()
    seg_lookup = load_segmentation(dataset_cfg, eval_cfg)
    videos = discover_videos(data_dir)

    all_rows = []
    for idx, video in enumerate(videos, start=1):
        all_rows.extend(
            compute_video_jitter_rows(
                video=video,
                models=models,
                data_dir=data_dir,
                dataset_cfg=dataset_cfg,
                eval_cfg=eval_cfg,
                seg_lookup=seg_lookup,
            )
        )
        if idx % 10 == 0 or idx == len(videos):
            print(
                f"[{idx}/{len(videos)}] processed {video['stem']} -> {len(all_rows):,} jitter rows",
                flush=True,
            )

    jitter_df = pd.DataFrame(all_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    jitter_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}", flush=True)

    summary = (
        jitter_df.groupby("model")["keypoint_jitter"]
        .agg(mean="mean", median="median", p90=lambda s: s.quantile(0.9), n_rows="size")
        .reset_index()
        .sort_values("mean")
    )
    print()
    print(summary.to_string(index=False), flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute true keypoint-displacement jitter.")
    parser.add_argument("--dataset", default="rehab24", choices=["rehab24", "rehab24_all"])
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
