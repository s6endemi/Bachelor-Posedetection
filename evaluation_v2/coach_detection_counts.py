"""
Reproducible detection-count analysis for the 5 coach extreme-case videos.

This script samples every Nth frame from the full video and records how many
persons each model detects before any person-selection logic is applied.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_evaluation.estimators.mediapipe_estimator import MediaPipeEstimator
from src.pose_evaluation.estimators.movenet_multipose_estimator import MoveNetMultiPoseEstimator
from src.pose_evaluation.estimators.yolo_estimator import YOLOPoseEstimator
from src.pose_evaluation.inference.data_loader import DataLoader, VideoSample


COACH_VIDEO_CONFIG = [
    {"video_id": "PM_010", "exercise": "Ex3", "camera": "c17"},
    {"video_id": "PM_011", "exercise": "Ex3", "camera": "c17"},
    {"video_id": "PM_108", "exercise": "Ex3", "camera": "c17"},
    {"video_id": "PM_119", "exercise": "Ex3", "camera": "c17"},
    {"video_id": "PM_121", "exercise": "Ex3", "camera": "c17"},
]


@dataclass
class DetectionCounterConfig:
    sample_step: int = 10
    mediapipe_min_detection_confidence: float = 0.1
    mediapipe_num_poses: int = 5
    movenet_score_threshold: float = 0.1
    yolo_box_confidence_threshold: float = 0.3


MODEL_SPECS = {
    "MediaPipe": {"prefix": "mediapipe", "counter_cls": None},
    "MoveNet": {"prefix": "movenet", "counter_cls": None},
    "YOLOv8": {"prefix": "yolo", "counter_cls": None},
}


class MediaPipeRawCounter:
    def __init__(self, config: DetectionCounterConfig):
        self.estimator = MediaPipeEstimator(
            model_complexity=1,
            min_detection_confidence=config.mediapipe_min_detection_confidence,
        )
        self.estimator._load_model()
        self.config = config

    def count(self, frame: np.ndarray) -> int:
        import mediapipe as mp

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.estimator._landmarker.detect(mp_image)
        if not results.pose_landmarks:
            return 0
        return len(results.pose_landmarks)


class MoveNetRawCounter:
    def __init__(self, config: DetectionCounterConfig):
        tfhub_cache = PROJECT_ROOT / ".tfhub_cache"
        tfhub_cache.mkdir(exist_ok=True)
        os.environ["TFHUB_CACHE_DIR"] = str(tfhub_cache)
        self.estimator = MoveNetMultiPoseEstimator()
        self.estimator._load_model()
        self.config = config

    def count(self, frame: np.ndarray) -> int:
        import tensorflow as tf

        h, w = frame.shape[:2]
        target_size = self.estimator._input_size
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb_frame, (new_w, new_h))

        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        pad_y = (target_size - new_h) // 2
        pad_x = (target_size - new_w) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        input_tensor = tf.cast(padded, dtype=tf.int32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        outputs = self.estimator._movenet(input_tensor)
        keypoints_with_scores = outputs["output_0"].numpy()[0]

        scores = keypoints_with_scores[:, 55]
        bboxes = keypoints_with_scores[:, 51:55]
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        valid = (scores > self.config.movenet_score_threshold) & (areas > 0)
        return int(np.sum(valid))


class YoloRawCounter:
    def __init__(self, config: DetectionCounterConfig):
        self.estimator = YOLOPoseEstimator(model_size="n")
        self.estimator._load_model()
        self.config = config

    def count(self, frame: np.ndarray) -> int:
        results = self.estimator._model(frame, verbose=False)
        if not results or results[0].boxes is None:
            return 0
        boxes = results[0].boxes
        if boxes.conf is None:
            return len(boxes)
        scores = boxes.conf.cpu().numpy()
        return int(np.sum(scores > self.config.yolo_box_confidence_threshold))


MODEL_SPECS["MediaPipe"]["counter_cls"] = MediaPipeRawCounter
MODEL_SPECS["MoveNet"]["counter_cls"] = MoveNetRawCounter
MODEL_SPECS["YOLOv8"]["counter_cls"] = YoloRawCounter


def discover_coach_samples() -> list[VideoSample]:
    loader = DataLoader(PROJECT_ROOT / "data")
    samples = loader.discover_samples()
    selected = []
    expected = {(cfg["video_id"], cfg["exercise"], cfg["camera"]) for cfg in COACH_VIDEO_CONFIG}
    for sample in samples:
        key = (sample.subject_id, sample.exercise, sample.camera)
        if key in expected:
            selected.append(sample)
    selected.sort(key=lambda s: (s.subject_id, s.exercise, s.camera))
    return selected


def categorize_count(count: int) -> str:
    if count <= 0:
        return "0p"
    if count == 1:
        return "1p"
    return "2p_plus"


def summarize_single_model_counts(frame_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    model_prefix = MODEL_SPECS[model_name]["prefix"]
    rows = []
    for video_id, group in frame_df.groupby("video_id"):
        counts = group["detected_person_count"]
        categories = counts.map(categorize_count)
        sample_frames = len(group)
        rows.append(
            {
                "video_id": video_id,
                "sampled_frames": sample_frames,
                f"{model_prefix}_0p": int((categories == "0p").sum()),
                f"{model_prefix}_1p": int((categories == "1p").sum()),
                f"{model_prefix}_2p_plus": int((categories == "2p_plus").sum()),
                f"{model_prefix}_0p_pct": float((categories == "0p").mean() * 100),
                f"{model_prefix}_1p_pct": float((categories == "1p").mean() * 100),
                f"{model_prefix}_2p_plus_pct": float((categories == "2p_plus").mean() * 100),
            }
        )

    total_counts = frame_df["detected_person_count"]
    total_categories = total_counts.map(categorize_count)
    rows.append(
        {
            "video_id": "TOTAL",
            "sampled_frames": len(frame_df),
            f"{model_prefix}_0p": int((total_categories == "0p").sum()),
            f"{model_prefix}_1p": int((total_categories == "1p").sum()),
            f"{model_prefix}_2p_plus": int((total_categories == "2p_plus").sum()),
            f"{model_prefix}_0p_pct": float((total_categories == "0p").mean() * 100),
            f"{model_prefix}_1p_pct": float((total_categories == "1p").mean() * 100),
            f"{model_prefix}_2p_plus_pct": float((total_categories == "2p_plus").mean() * 100),
        }
    )
    return pd.DataFrame(rows)


def run_single_model_analysis(
    model_name: str,
    config: DetectionCounterConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = discover_coach_samples()
    if len(samples) != len(COACH_VIDEO_CONFIG):
        raise RuntimeError(
            f"Expected {len(COACH_VIDEO_CONFIG)} coach samples, found {len(samples)}."
        )

    counter_cls = MODEL_SPECS[model_name]["counter_cls"]
    counter = counter_cls(config)

    frame_rows = []
    for sample in samples:
        cap = cv2.VideoCapture(str(sample.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {sample.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        sampled_frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % config.sample_step != 0:
                frame_idx += 1
                continue

            frame_rows.append(
                {
                    "video_id": sample.subject_id,
                    "exercise": sample.exercise,
                    "camera": sample.camera,
                    "video_path": str(sample.video_path),
                    "sampled_video_frame_idx": frame_idx,
                    "detected_person_count": counter.count(frame),
                }
            )
            sampled_frames += 1
            frame_idx += 1

        cap.release()
        print(
            f"[coach-counts:{model_name}] {sample.subject_id}-{sample.camera} "
            f"sampled {sampled_frames} / {total_frames} video frames",
            flush=True,
        )

    frame_df = pd.DataFrame(frame_rows)
    summary_df = summarize_single_model_counts(frame_df, model_name)
    return frame_df, summary_df


def get_tmp_paths(model_name: str) -> tuple[Path, Path]:
    out_dir = PROJECT_ROOT / "evaluation_v2" / "results" / "coach_detection_counts_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = MODEL_SPECS[model_name]["prefix"]
    return out_dir / f"{slug}_frames.csv", out_dir / f"{slug}_summary.csv"


def write_single_model_outputs(
    model_name: str,
    frame_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    frame_path, summary_path = get_tmp_paths(model_name)
    frame_df.to_csv(frame_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {frame_path}", flush=True)
    print(f"Saved: {summary_path}", flush=True)


def merge_model_outputs(config: DetectionCounterConfig) -> None:
    out_dir = PROJECT_ROOT / "evaluation_v2" / "results"
    combined_frame_path = out_dir / "coach_detection_counts_frames.csv"
    combined_summary_path = out_dir / "coach_detection_counts_summary.csv"
    metadata_path = out_dir / "coach_detection_counts_metadata.json"

    merged_frames = None
    merged_summary = None
    for model_name, spec in MODEL_SPECS.items():
        frame_path, summary_path = get_tmp_paths(model_name)
        frame_df = pd.read_csv(frame_path).rename(
            columns={"detected_person_count": f"{spec['prefix']}_count"}
        )
        summary_df = pd.read_csv(summary_path)

        frame_keep = [
            "video_id",
            "exercise",
            "camera",
            "video_path",
            "sampled_video_frame_idx",
            f"{spec['prefix']}_count",
        ]
        merged_frames = (
            frame_df[frame_keep]
            if merged_frames is None
            else merged_frames.merge(
                frame_df[frame_keep],
                on=["video_id", "exercise", "camera", "video_path", "sampled_video_frame_idx"],
                how="outer",
            )
        )

        merged_summary = (
            summary_df
            if merged_summary is None
            else merged_summary.merge(summary_df, on=["video_id", "sampled_frames"], how="outer")
        )

    merged_frames.to_csv(combined_frame_path, index=False)
    merged_summary.to_csv(combined_summary_path, index=False)

    metadata = {
        "sample_step": config.sample_step,
        "coach_videos": COACH_VIDEO_CONFIG,
        "thresholds": {
            "mediapipe_min_detection_confidence": config.mediapipe_min_detection_confidence,
            "mediapipe_num_poses": config.mediapipe_num_poses,
            "movenet_score_threshold": config.movenet_score_threshold,
            "yolo_box_confidence_threshold": config.yolo_box_confidence_threshold,
        },
        "frame_csv": str(combined_frame_path),
        "summary_csv": str(combined_summary_path),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {combined_frame_path}", flush=True)
    print(f"Saved: {combined_summary_path}", flush=True)
    print(f"Saved: {metadata_path}", flush=True)
    print()
    print("| Video | Sampled Frames | MediaPipe 1p | MediaPipe 2+ | MoveNet 1p | MoveNet 2+ | YOLO 1p | YOLO 2+ |")
    print("|-------|---------------:|-------------:|-------------:|-----------:|-----------:|--------:|--------:|")
    for _, row in merged_summary.iterrows():
        print(
            f"| {row['video_id']} | {int(row['sampled_frames'])} | "
            f"{row['mediapipe_1p_pct']:.0f}% | {row['mediapipe_2p_plus_pct']:.0f}% | "
            f"{row['movenet_1p_pct']:.0f}% | {row['movenet_2p_plus_pct']:.0f}% | "
            f"{row['yolo_1p_pct']:.0f}% | {row['yolo_2p_plus_pct']:.0f}% |"
        )


def run_orchestrated(config: DetectionCounterConfig) -> None:
    for model_name in MODEL_SPECS:
        print(f"[coach-counts] launching subprocess for {model_name}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__)),
                "--model",
                model_name,
                "--sample-step",
                str(config.sample_step),
            ],
            check=True,
            cwd=str(PROJECT_ROOT),
        )
    merge_model_outputs(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coach detection-count analysis")
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--sample-step", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DetectionCounterConfig(sample_step=args.sample_step)
    if args.model:
        frame_df, summary_df = run_single_model_analysis(args.model, config)
        write_single_model_outputs(args.model, frame_df, summary_df)
    else:
        run_orchestrated(config)


if __name__ == "__main__":
    main()
