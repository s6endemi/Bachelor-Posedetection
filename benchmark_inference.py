"""
Inference Time Benchmark for Pose Estimation Models.

Measures per-frame inference time for each model independently.
Results are saved separately from accuracy evaluation (methodisch sauber).

Usage:
    python benchmark_inference.py                    # Default: 3 videos, 500 frames each
    python benchmark_inference.py --num-videos 5     # More videos
    python benchmark_inference.py --frames 1000      # More frames per video
    python benchmark_inference.py --warmup 100       # Longer warmup
"""

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.pose_evaluation.estimators import (
    MediaPipeEstimator,
    YOLOPoseEstimator,
)
from src.pose_evaluation.estimators.base import PoseEstimator, Keypoint
from src.pose_evaluation.inference import DataLoader


class MoveNetTFLiteEstimator(PoseEstimator):
    """MoveNet MultiPose via TFLite (deployment-optimized runtime)."""

    MOVENET_TO_COCO = {i: i for i in range(17)}

    def __init__(self, model_path: str = "models/movenet_multipose_lightning_fp16.tflite"):
        self._model_path = model_path
        self._interpreter = None
        self._input_size = 256

    def _load_model(self):
        if self._interpreter is None:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(model_path=self._model_path)
            # MultiPose has dynamic input shape - resize to 256x256
            input_details = self._interpreter.get_input_details()
            self._interpreter.resize_tensor_input(
                input_details[0]['index'], [1, self._input_size, self._input_size, 3]
            )
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

    def predict(self, frame: np.ndarray) -> list[Keypoint]:
        import cv2

        self._load_model()

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Same preprocessing as TF Hub version: resize with padding
        target_size = self._input_size
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb_frame, (new_w, new_h))

        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        pad_y = (target_size - new_h) // 2
        pad_x = (target_size - new_w) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # TFLite expects uint8 input with batch dimension
        input_tensor = np.expand_dims(padded, axis=0).astype(np.uint8)

        self._interpreter.set_tensor(self._input_details[0]['index'], input_tensor)
        self._interpreter.invoke()

        # Output: [1, 6, 56]
        output = self._interpreter.get_tensor(self._output_details[0]['index'])
        keypoints_with_scores = output[0]  # Shape: (6, 56)

        # Same person selection as TF Hub version: largest bounding box
        best_idx = 0
        best_area = 0
        for i in range(6):
            person = keypoints_with_scores[i]
            ymin, xmin, ymax, xmax = person[51:55]
            area = (xmax - xmin) * (ymax - ymin)
            score = person[55]
            if score > 0.1 and area > best_area:
                best_area = area
                best_idx = i

        if best_area == 0:
            return [
                Keypoint(x=0.0, y=0.0, confidence=0.0, name=name)
                for name in self.COCO_KEYPOINTS
            ]

        best_person = keypoints_with_scores[best_idx]

        native_keypoints = []
        for idx in range(17):
            y_norm = best_person[idx * 3]
            x_norm = best_person[idx * 3 + 1]
            confidence = best_person[idx * 3 + 2]

            x_padded = x_norm * target_size
            y_padded = y_norm * target_size
            x_orig = (x_padded - pad_x) / scale
            y_orig = (y_padded - pad_y) / scale

            native_keypoints.append(Keypoint(
                x=float(x_orig), y=float(y_orig),
                confidence=float(confidence),
                name=self.COCO_KEYPOINTS[idx]
            ))

        return self.map_to_coco(native_keypoints)

    def get_model_name(self) -> str:
        return "MoveNet (multipose)"

    @property
    def native_keypoint_mapping(self) -> dict[int, int]:
        return self.MOVENET_TO_COCO


def get_hardware_info() -> dict:
    """Collect hardware information for reproducibility."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }

    # GPU info (if available)
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
            info["cuda_version"] = torch.version.cuda
        else:
            info["gpu"] = "None (CPU only)"
    except ImportError:
        info["gpu"] = "torch not available"

    # RAM
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        info["ram_gb"] = "psutil not available"

    return info


def benchmark_model(estimator, video_paths: list[Path], frames_per_video: int,
                    warmup_frames: int) -> dict:
    """
    Benchmark a single model on multiple videos.

    Args:
        estimator: PoseEstimator instance
        video_paths: List of video files to benchmark on
        frames_per_video: Number of frames to measure per video (after warmup)
        warmup_frames: Number of warmup frames to discard

    Returns:
        Dict with timing statistics
    """
    model_name = estimator.get_model_name()
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    all_times = []
    video_stats = []

    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [WARN] Cannot open {video_path.name}, skipping")
            continue

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        needed = warmup_frames + frames_per_video
        actual_measure = min(frames_per_video, max(0, total_video_frames - warmup_frames))

        print(f"\n  Video: {video_path.name} ({total_video_frames} frames, {width}x{height})")
        print(f"  Warmup: {warmup_frames} frames, Measure: {actual_measure} frames")

        frame_times = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Inference + timing
            t_start = time.perf_counter()
            _ = estimator.predict(frame)
            t_end = time.perf_counter()

            elapsed_ms = (t_end - t_start) * 1000

            # Skip warmup frames
            if frame_idx >= warmup_frames:
                frame_times.append(elapsed_ms)

            frame_idx += 1

            # Stop after enough frames
            if len(frame_times) >= frames_per_video:
                break

        cap.release()

        if frame_times:
            times_arr = np.array(frame_times)
            video_stat = {
                "video": video_path.name,
                "resolution": f"{width}x{height}",
                "n_frames": len(frame_times),
                "mean_ms": float(np.mean(times_arr)),
                "std_ms": float(np.std(times_arr)),
                "median_ms": float(np.median(times_arr)),
                "min_ms": float(np.min(times_arr)),
                "max_ms": float(np.max(times_arr)),
                "p95_ms": float(np.percentile(times_arr, 95)),
                "p99_ms": float(np.percentile(times_arr, 99)),
            }
            video_stats.append(video_stat)
            all_times.extend(frame_times)

            print(f"  -> Mean: {video_stat['mean_ms']:.1f}ms | "
                  f"Median: {video_stat['median_ms']:.1f}ms | "
                  f"Std: {video_stat['std_ms']:.1f}ms | "
                  f"P95: {video_stat['p95_ms']:.1f}ms")

    # Aggregate across all videos
    if all_times:
        all_arr = np.array(all_times)
        aggregate = {
            "model": model_name,
            "total_frames": len(all_times),
            "mean_ms": float(np.mean(all_arr)),
            "std_ms": float(np.std(all_arr)),
            "median_ms": float(np.median(all_arr)),
            "min_ms": float(np.min(all_arr)),
            "max_ms": float(np.max(all_arr)),
            "p95_ms": float(np.percentile(all_arr, 95)),
            "p99_ms": float(np.percentile(all_arr, 99)),
            "fps": float(1000 / np.mean(all_arr)),
        }

        print(f"\n  AGGREGATE ({aggregate['total_frames']} frames):")
        print(f"  Mean: {aggregate['mean_ms']:.1f}ms | "
              f"FPS: {aggregate['fps']:.1f} | "
              f"Median: {aggregate['median_ms']:.1f}ms")
    else:
        aggregate = {"model": model_name, "total_frames": 0}

    return {
        "aggregate": aggregate,
        "per_video": video_stats,
        "raw_times_ms": all_times,
    }


def select_benchmark_videos(data_root: Path, num_videos: int) -> list[Path]:
    """Select a diverse set of videos for benchmarking."""
    loader = DataLoader(data_root)
    samples = loader.discover_samples()

    if not samples:
        raise RuntimeError("No videos found in data/videos/")

    # Pick from different exercises for diversity
    by_exercise = {}
    for s in samples:
        by_exercise.setdefault(s.exercise, []).append(s)

    selected = []
    exercises = sorted(by_exercise.keys())
    idx = 0
    while len(selected) < num_videos and idx < len(samples):
        for ex in exercises:
            if len(selected) >= num_videos:
                break
            ex_samples = by_exercise[ex]
            # Pick c17 cameras (not transposed) for consistency
            c17 = [s for s in ex_samples if s.camera == "c17"]
            pick_from = c17 if c17 else ex_samples
            if idx < len(pick_from):
                selected.append(pick_from[idx].video_path)
        idx += 1

    print(f"Selected {len(selected)} videos for benchmark:")
    for v in selected:
        print(f"  - {v.parent.name}/{v.name}")

    return selected


def main():
    parser = argparse.ArgumentParser(description="Benchmark Inference Time for Pose Models")
    parser.add_argument("--num-videos", type=int, default=3,
                        help="Number of videos to benchmark on (default: 3)")
    parser.add_argument("--frames", type=int, default=500,
                        help="Frames to measure per video after warmup (default: 500)")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Warmup frames to discard per video (default: 50)")
    parser.add_argument("--yolo-size", type=str, default="n",
                        help="YOLOv8 model size, must match evaluation (default: n)")
    args = parser.parse_args()

    print("=" * 60)
    print("INFERENCE TIME BENCHMARK")
    print("=" * 60)
    print(f"Config: {args.num_videos} videos, {args.frames} frames/video, "
          f"{args.warmup} warmup frames")
    print()

    # Hardware info
    hw_info = get_hardware_info()
    print("Hardware:")
    for k, v in hw_info.items():
        print(f"  {k}: {v}")
    print()

    # Select videos
    data_root = Path("data")
    video_paths = select_benchmark_videos(data_root, args.num_videos)
    print()

    # Models (same config as main evaluation)
    models = [
        ("MediaPipe", MediaPipeEstimator(model_complexity=1)),
        ("MoveNet", MoveNetTFLiteEstimator()),
        ("YOLO", YOLOPoseEstimator(model_size=args.yolo_size)),
    ]

    # Benchmark each model separately
    results = {}
    for name, estimator in models:
        results[name] = benchmark_model(
            estimator, video_paths, args.frames, args.warmup
        )

    # === Save Results ===
    output_dir = Path("analysis/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary CSV (one row per model - like temporal_jitter.csv)
    summary_rows = []
    for name in results:
        agg = results[name]["aggregate"]
        if agg.get("total_frames", 0) > 0:
            summary_rows.append(agg)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "inference_benchmark.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    # 2. Metadata JSON (hardware, config, per-video details for reproducibility)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "hardware": hw_info,
        "config": {
            "num_videos": args.num_videos,
            "frames_per_video": args.frames,
            "warmup_frames": args.warmup,
            "yolo_size": args.yolo_size,
        },
        "videos": [str(v) for v in video_paths],
        "results": {
            name: {
                "aggregate": results[name]["aggregate"],
                "per_video": results[name]["per_video"],
            }
            for name in results
        },
    }
    meta_path = output_dir / "inference_benchmark_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    # === Print Final Summary ===
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Mean (ms)':>10} {'Median':>10} {'FPS':>8} {'P95':>10}")
    print("-" * 63)
    for row in summary_rows:
        print(f"{row['model']:<25} {row['mean_ms']:>10.1f} {row['median_ms']:>10.1f} "
              f"{row['fps']:>8.1f} {row['p95_ms']:>10.1f}")
    print()


if __name__ == "__main__":
    main()
