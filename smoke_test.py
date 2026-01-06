"""
Smoke Test: Prüft ob alle 3 Pose Estimation Modelle funktionieren.
"""

import sys
sys.path.insert(0, "src")

import cv2
import numpy as np
from pathlib import Path

def test_video_loading():
    """Test 1: Kann ein Video geladen werden?"""
    video_path = Path("data/videos/Ex1/PM_000-Camera17-30fps.mp4")

    if not video_path.exists():
        print(f"[FAIL] Video nicht gefunden: {video_path}")
        return None

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[FAIL] Konnte Frame nicht lesen")
        return None

    print(f"[OK] Video geladen: {frame.shape}")
    return frame


def test_mediapipe(frame):
    """Test 2: MediaPipe Pose"""
    try:
        from pose_evaluation.estimators import MediaPipeEstimator

        estimator = MediaPipeEstimator(model_complexity=1)  # complexity=1 für schnelleren Test
        keypoints = estimator.predict(frame)

        # Check ob wir 17 Keypoints haben
        if len(keypoints) != 17:
            print(f"[FAIL] MediaPipe: Erwartet 17 Keypoints, bekommen {len(keypoints)}")
            return False

        # Check ob mindestens einige Keypoints confidence > 0 haben
        valid_kps = sum(1 for kp in keypoints if kp.confidence > 0.1)
        print(f"[OK] MediaPipe: {valid_kps}/17 Keypoints erkannt")
        return True

    except Exception as e:
        print(f"[FAIL] MediaPipe: {e}")
        return False


def test_movenet(frame):
    """Test 3: MoveNet"""
    try:
        from pose_evaluation.estimators import MoveNetEstimator

        estimator = MoveNetEstimator(model_name="thunder")
        keypoints = estimator.predict(frame)

        if len(keypoints) != 17:
            print(f"[FAIL] MoveNet: Erwartet 17 Keypoints, bekommen {len(keypoints)}")
            return False

        valid_kps = sum(1 for kp in keypoints if kp.confidence > 0.1)
        print(f"[OK] MoveNet: {valid_kps}/17 Keypoints erkannt")
        return True

    except Exception as e:
        print(f"[FAIL] MoveNet: {e}")
        return False


def test_yolo(frame):
    """Test 4: YOLOv8-Pose"""
    try:
        from pose_evaluation.estimators import YOLOPoseEstimator

        estimator = YOLOPoseEstimator(model_size="n")  # nano für schnelleren Test
        keypoints = estimator.predict(frame)

        if len(keypoints) != 17:
            print(f"[FAIL] YOLO: Erwartet 17 Keypoints, bekommen {len(keypoints)}")
            return False

        valid_kps = sum(1 for kp in keypoints if kp.confidence > 0.1)
        print(f"[OK] YOLO: {valid_kps}/17 Keypoints erkannt")
        return True

    except Exception as e:
        print(f"[FAIL] YOLO: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("SMOKE TEST - Pose Estimation Pipeline")
    print("=" * 50)

    # Test 1: Video laden
    print("\n[1/4] Video laden...")
    frame = test_video_loading()

    if frame is None:
        print("\nAbbruch: Kein Frame zum Testen")
        sys.exit(1)

    # Test 2: MediaPipe
    print("\n[2/4] MediaPipe testen...")
    mp_ok = test_mediapipe(frame)

    # Test 3: MoveNet
    print("\n[3/4] MoveNet testen...")
    mn_ok = test_movenet(frame)

    # Test 4: YOLO
    print("\n[4/4] YOLOv8-Pose testen...")
    yolo_ok = test_yolo(frame)

    # Summary
    print("\n" + "=" * 50)
    print("ZUSAMMENFASSUNG")
    print("=" * 50)

    results = {
        "Video": frame is not None,
        "MediaPipe": mp_ok,
        "MoveNet": mn_ok,
        "YOLO": yolo_ok
    }

    for name, ok in results.items():
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}")

    if all(results.values()):
        print("\nAlle Tests bestanden!")
        sys.exit(0)
    else:
        print("\nEinige Tests fehlgeschlagen.")
        sys.exit(1)
