"""
REHAB24-6 Variant Inference — runs one model variant on all videos.

Saves predictions to data/predictions_variants/{variant_name}/{Ex}/{video}.npz
Same format as original .npz: (N, 17, 3) per frame.

Usage:
    python evaluation_v2/rehab_variant_inference.py --model MediaPipe_Lite --test
    python evaluation_v2/rehab_variant_inference.py --model YOLOv8m
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_model(model_name: str):
    """Load a model variant by name."""
    if model_name == "MediaPipe_Lite":
        from src.pose_evaluation.estimators.mediapipe_estimator import MediaPipeEstimator
        return MediaPipeEstimator(model_complexity=0, min_detection_confidence=0.1)
    elif model_name == "MediaPipe_Heavy":
        from src.pose_evaluation.estimators.mediapipe_estimator import MediaPipeEstimator
        return MediaPipeEstimator(model_complexity=2, min_detection_confidence=0.1)
    elif model_name == "MoveNet_SP_Lightning":
        from src.pose_evaluation.estimators.movenet_estimator import MoveNetEstimator
        return MoveNetEstimator(model_name="lightning")
    elif model_name == "MoveNet_SP_Thunder":
        from src.pose_evaluation.estimators.movenet_estimator import MoveNetEstimator
        return MoveNetEstimator(model_name="thunder")
    elif model_name == "YOLOv8s":
        from src.pose_evaluation.estimators.yolo_estimator import YOLOPoseEstimator
        return YOLOPoseEstimator(model_size="s")
    elif model_name == "YOLOv8m":
        from src.pose_evaluation.estimators.yolo_estimator import YOLOPoseEstimator
        return YOLOPoseEstimator(model_size="m")
    else:
        raise ValueError(f"Unknown variant: {model_name}")


def predict_frame(estimator, frame: np.ndarray) -> np.ndarray:
    """Run model, return (17, 3) array."""
    try:
        keypoints = estimator.predict(frame)
        if keypoints is None or len(keypoints) == 0:
            return np.zeros((17, 3))
        result = np.zeros((17, 3))
        for i, kp in enumerate(keypoints[:17]):
            result[i] = [kp.x, kp.y, kp.confidence]
        return result
    except Exception:
        return np.zeros((17, 3))


def find_videos(data_dir: Path) -> list[dict]:
    """Find all videos that have existing .npz predictions."""
    videos = []
    pred_dir = data_dir / "predictions"
    video_dir = data_dir / "videos"

    for ex_dir in sorted(pred_dir.iterdir()):
        if not ex_dir.is_dir():
            continue
        exercise = ex_dir.name

        for npz_file in sorted(ex_dir.glob("*.npz")):
            stem = npz_file.stem  # PM_000-c17
            parts = stem.rsplit("-", 1)
            if len(parts) != 2:
                continue
            video_id, camera = parts

            # Find matching video file
            vid_dir = video_dir / exercise
            if not vid_dir.exists():
                continue

            # Try common naming patterns
            vid_path = None
            for pattern in [
                f"{video_id}-Camera{camera[1:]}-30fps.mp4",
                f"{video_id}-Camera{camera[1:]}-30fps-transposed.mp4",
            ]:
                candidate = vid_dir / pattern
                if candidate.exists():
                    vid_path = candidate
                    break

            if vid_path is None:
                # Try glob
                matches = list(vid_dir.glob(f"{video_id}-Camera{camera[1:]}*"))
                if matches:
                    vid_path = matches[0]

            videos.append({
                "npz_path": npz_file,
                "video_path": vid_path,
                "exercise": exercise,
                "video_id": video_id,
                "camera": camera,
                "stem": stem,
            })

    return videos


def process_video(
    video: dict,
    estimator,
    model_name: str,
    output_dir: Path,
    frame_step: int = 3,
    skip_existing: bool = False,
) -> dict:
    """Process one video with one model variant."""
    # Skip if already processed
    out_path = output_dir / video["exercise"] / f"{video['stem']}.npz"
    if skip_existing and out_path.exists():
        return {"status": "skipped", "frames": 0, "mean_ms": 0, "fps": 0}

    # Load existing .npz for rotation angles and frame count
    existing = np.load(video["npz_path"])
    n_frames = int(existing["num_frames"])
    rotation_angles = existing["rotation_angles"]

    # Open video
    if video["video_path"] is None or not video["video_path"].exists():
        return {"status": "no_video", "frames": 0}

    cap = cv2.VideoCapture(str(video["video_path"]))
    if not cap.isOpened():
        return {"status": "cant_open", "frames": 0}

    predictions = np.zeros((n_frames, 17, 3), dtype=np.float64)
    frame_times = []

    # Sequential read (4x faster than seeking)
    frame_counter = 0
    pred_idx = 0
    while pred_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_counter % frame_step == 0:
            t0 = time.perf_counter()
            predictions[pred_idx] = predict_frame(estimator, frame)
            t1 = time.perf_counter()
            frame_times.append(t1 - t0)
            pred_idx += 1
        frame_counter += 1

    cap.release()

    # Save .npz
    out_dir = output_dir / video["exercise"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video['stem']}.npz"

    pred_key = f"pred_{model_name}"
    np.savez(
        out_path,
        rotation_angles=rotation_angles[:len(predictions)],
        num_frames=len(predictions),
        **{pred_key: predictions},
    )

    mean_ms = np.mean(frame_times) * 1000 if frame_times else 0
    return {
        "status": "ok",
        "frames": len(predictions),
        "mean_ms": mean_ms,
        "fps": 1000 / mean_ms if mean_ms > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model variant name")
    parser.add_argument("--test", action="store_true", help="Only 3 videos")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos with existing .npz")
    parser.add_argument("--frame-step", type=int, default=3)
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data"
    output_dir = data_dir / "predictions_variants" / args.model

    print(f"=== REHAB24-6 Variant Inference: {args.model} ===")
    print(f"  Output: {output_dir}")

    # Find videos
    videos = find_videos(data_dir)
    if args.test:
        videos = videos[:3]
    print(f"  Videos: {len(videos)}")

    # Load model
    print(f"  Loading {args.model}...")
    estimator = load_model(args.model)
    print(f"  Model loaded.\n")

    # Process
    total_frames = 0
    all_fps = []
    t_start = time.time()

    for i, video in enumerate(videos):
        result = process_video(video, estimator, args.model, output_dir, args.frame_step, args.skip_existing)

        total_frames += result["frames"]
        if result["fps"] > 0:
            all_fps.append(result["fps"])

        elapsed = time.time() - t_start
        vps = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(videos) - i - 1) / vps / 60 if vps > 0 else 0

        if (i + 1) % 10 == 0 or (i + 1) == len(videos) or args.test:
            status = result["status"]
            fps_str = f"{result['fps']:.1f} FPS" if result["fps"] > 0 else "N/A"
            print(f"  [{i+1}/{len(videos)}] {video['stem']} -> {status}, "
                  f"{result['frames']} frames, {fps_str}, ETA {eta:.0f}min")

    # Summary
    elapsed = time.time() - t_start
    mean_fps = np.mean(all_fps) if all_fps else 0
    print(f"\n=== Done: {args.model} ===")
    print(f"  Total: {total_frames:,} frames in {elapsed/60:.1f} min")
    print(f"  Speed: {mean_fps:.1f} FPS (mean)")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
