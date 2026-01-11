"""
Hauptskript zum Ausfuehren der Inference-Pipeline.

Usage:
    python run_inference.py                    # Alle Videos
    python run_inference.py --test             # Schnelltest (1 Video, 10 Frames)
    python run_inference.py --exercise Ex1    # Nur Exercise 1
"""

import argparse
from pathlib import Path

from src.pose_evaluation.estimators import (
    MediaPipeEstimator,
    MoveNetMultiPoseEstimator,
    YOLOPoseEstimator
)
from src.pose_evaluation.inference import InferencePipeline


def main():
    parser = argparse.ArgumentParser(description="Run Pose Estimation Inference")
    parser.add_argument("--test", action="store_true", help="Quick test mode (1 video, 10 frames)")
    parser.add_argument("--exercise", type=str, help="Only process specific exercise (Ex1-Ex6)")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos to process")
    parser.add_argument("--max-frames", type=int, help="Maximum frames per video")
    parser.add_argument("--yolo-size", type=str, default="m", choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size (default: m)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Process every Nth frame (default: 1 = all frames, 3 = every 3rd)")

    args = parser.parse_args()

    # Test-Modus
    if args.test:
        args.max_videos = 1
        args.max_frames = 10
        args.yolo_size = "n"  # Nano fuer schnellen Test
        print("=== TEST MODE ===")
        print()

    # Estimators initialisieren (3 finale Modelle)
    print("Loading models...")
    estimators = [
        MediaPipeEstimator(model_complexity=1),      # Full (good balance of speed/accuracy)
        MoveNetMultiPoseEstimator(),                 # MultiPose, BBox-Selection
        YOLOPoseEstimator(model_size=args.yolo_size), # BBox-Selection
    ]
    print(f"Models: {[e.get_model_name() for e in estimators]}")

    # Pipeline
    pipeline = InferencePipeline(
        estimators=estimators,
        data_root=Path("data"),
        output_dir=Path("data/predictions")
    )

    # Exercises filtern
    exercises = None
    if args.exercise:
        exercises = [args.exercise]

    # Ausfuehren
    pipeline.run(
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames,
        exercises=exercises,
        frame_step=args.frame_step
    )


if __name__ == "__main__":
    main()
