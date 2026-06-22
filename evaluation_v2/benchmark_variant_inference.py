"""
Inference benchmark for the 6 non-baseline model variants.

Uses the same benchmark protocol as `benchmark_inference.py`:
- same benchmark video selection
- same warmup handling
- same per-frame timing logic
- sequential execution to avoid resource contention

Outputs:
- evaluation_v2/results/inference_benchmark_variants.csv
- evaluation_v2/results/inference_benchmark_variants_meta.json
- evaluation_v2/results/inference_benchmark_all.csv
- evaluation_v2/results/inference_benchmark_all_meta.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_inference import benchmark_model, get_hardware_info, select_benchmark_videos
from src.pose_evaluation.estimators import MediaPipeEstimator, YOLOPoseEstimator
from src.pose_evaluation.estimators.base import Keypoint


class NamedEstimator:
    """Lightweight wrapper that overrides the display name used in the benchmark outputs."""

    def __init__(self, estimator, name: str):
        self.estimator = estimator
        self.name = name

    def predict(self, frame):
        return self.estimator.predict(frame)

    def get_model_name(self) -> str:
        return self.name


class MoveNetSinglePoseTFLiteEstimator:
    """TFLite single-pose MoveNet for benchmark use."""

    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    def __init__(self, model_path: Path, input_size: int, name: str):
        self.model_path = str(model_path)
        self.input_size = input_size
        self.name = name
        self._interpreter = None

    def _load_model(self):
        if self._interpreter is None:
            import tensorflow as tf

            self._interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

    def predict(self, frame):
        import cv2

        self._load_model()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size))
        input_tensor = np.expand_dims(resized.astype(np.uint8), axis=0)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
        self._interpreter.invoke()
        kps = self._interpreter.get_tensor(self._output_details[0]["index"])[0, 0]

        return [
            Keypoint(
                x=float(x * w),
                y=float(y * h),
                confidence=float(c),
                name=self.COCO_KEYPOINTS[i],
            )
            for i, (y, x, c) in enumerate(kps)
        ]

    def get_model_name(self) -> str:
        return self.name


def load_baseline_results() -> pd.DataFrame:
    baseline_path = PROJECT_ROOT / "analysis" / "results" / "inference_benchmark.csv"
    baseline = pd.read_csv(baseline_path).copy()
    name_map = {
        "MediaPipe (full)": "MediaPipe_Full",
        "MoveNet (multipose)": "MoveNet_MultiPose",
        "YOLOv8-Pose (n)": "YOLOv8_Nano",
    }
    baseline["model"] = baseline["model"].map(name_map)
    baseline["source"] = "analysis/results/inference_benchmark.csv"
    return baseline


def build_estimators(models_dir: Path) -> list[tuple[str, object]]:
    return [
        (
            "MediaPipe_Lite",
            NamedEstimator(
                MediaPipeEstimator(model_complexity=0, min_detection_confidence=0.1),
                "MediaPipe_Lite",
            ),
        ),
        (
            "MediaPipe_Heavy",
            NamedEstimator(
                MediaPipeEstimator(model_complexity=2, min_detection_confidence=0.1),
                "MediaPipe_Heavy",
            ),
        ),
        (
            "MoveNet_SP_Lightning",
            MoveNetSinglePoseTFLiteEstimator(
                models_dir / "movenet_singlepose_lightning_fp16.tflite",
                input_size=192,
                name="MoveNet_SP_Lightning",
            ),
        ),
        (
            "MoveNet_SP_Thunder",
            MoveNetSinglePoseTFLiteEstimator(
                models_dir / "movenet_singlepose_thunder_fp16.tflite",
                input_size=256,
                name="MoveNet_SP_Thunder",
            ),
        ),
        (
            "YOLOv8_Small",
            NamedEstimator(YOLOPoseEstimator(model_size="s"), "YOLOv8_Small"),
        ),
        (
            "YOLOv8_Medium",
            NamedEstimator(YOLOPoseEstimator(model_size="m"), "YOLOv8_Medium"),
        ),
    ]


def run_variant_benchmark(num_videos: int, frames: int, warmup: int) -> None:
    models_dir = PROJECT_ROOT / "models"
    missing = [
        path.name
        for path in [
            models_dir / "movenet_singlepose_lightning_fp16.tflite",
            models_dir / "movenet_singlepose_thunder_fp16.tflite",
            models_dir / "pose_landmarker_lite.task",
            models_dir / "pose_landmarker_heavy.task",
            models_dir / "yolov8s-pose.pt",
            models_dir / "yolov8m-pose.pt",
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required model files: {missing}")

    print("=" * 60)
    print("VARIANT INFERENCE BENCHMARK")
    print("=" * 60)
    print(f"Config: {num_videos} videos, {frames} frames/video, {warmup} warmup frames")
    print()

    hw_info = get_hardware_info()
    print("Hardware:")
    for key, value in hw_info.items():
        print(f"  {key}: {value}")
    print()

    video_paths = select_benchmark_videos(PROJECT_ROOT / "data", num_videos)
    print()

    results = {}
    for canonical_name, estimator in build_estimators(models_dir):
        results[canonical_name] = benchmark_model(estimator, video_paths, frames, warmup)

    output_dir = PROJECT_ROOT / "evaluation_v2" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_rows = []
    for canonical_name, result in results.items():
        aggregate = result["aggregate"]
        if aggregate.get("total_frames", 0) > 0:
            row = dict(aggregate)
            row["model"] = canonical_name
            row["source"] = "evaluation_v2/benchmark_variant_inference.py"
            variant_rows.append(row)

    variant_df = pd.DataFrame(variant_rows).sort_values("mean_ms")
    variant_csv = output_dir / "inference_benchmark_variants.csv"
    variant_df.to_csv(variant_csv, index=False)

    baseline_df = load_baseline_results()
    combined_df = pd.concat([baseline_df, variant_df], ignore_index=True)
    combined_df = combined_df.sort_values("mean_ms").reset_index(drop=True)
    combined_csv = output_dir / "inference_benchmark_all.csv"
    combined_df.to_csv(combined_csv, index=False)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "hardware": hw_info,
        "config": {
            "num_videos": num_videos,
            "frames_per_video": frames,
            "warmup_frames": warmup,
        },
        "videos": [str(path) for path in video_paths],
        "variant_results": {
            model: {
                "aggregate": result["aggregate"],
                "per_video": result["per_video"],
            }
            for model, result in results.items()
        },
    }
    variant_meta = output_dir / "inference_benchmark_variants_meta.json"
    with open(variant_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    combined_meta = output_dir / "inference_benchmark_all_meta.json"
    with open(combined_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                **metadata,
                "baseline_source": str(PROJECT_ROOT / "analysis" / "results" / "inference_benchmark.csv"),
                "combined_csv": str(combined_csv),
            },
            f,
            indent=2,
        )

    print(f"\nSaved: {variant_csv}")
    print(f"Saved: {variant_meta}")
    print(f"Saved: {combined_csv}")
    print(f"Saved: {combined_meta}")

    print(f"\n{'=' * 70}")
    print("COMBINED SUMMARY (ALL 9 MODELS)")
    print(f"{'=' * 70}")
    print(f"{'Model':<24} {'Mean (ms)':>10} {'Median':>10} {'FPS':>8} {'P95':>10}")
    print("-" * 66)
    for _, row in combined_df.iterrows():
        print(
            f"{row['model']:<24} {row['mean_ms']:>10.1f} {row['median_ms']:>10.1f} "
            f"{row['fps']:>8.1f} {row['p95_ms']:>10.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference time for 6 model variants.")
    parser.add_argument("--num-videos", type=int, default=3)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()
    run_variant_benchmark(args.num_videos, args.frames, args.warmup)


if __name__ == "__main__":
    main()
