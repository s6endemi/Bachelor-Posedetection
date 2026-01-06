"""
MediaPipe Pose Estimator Implementation.
Uses the new MediaPipe Tasks API (PoseLandmarker).
"""

import numpy as np
from pathlib import Path
from .base import PoseEstimator, Keypoint


class MediaPipeEstimator(PoseEstimator):
    """MediaPipe Pose Implementation (Top-Down Architektur)."""

    # MediaPipe hat 33 Keypoints, wir mappen nur die relevanten auf COCO
    MEDIAPIPE_TO_COCO = {
        0: 0,    # nose -> nose
        2: 1,    # left_eye_inner -> left_eye (Approximation)
        5: 2,    # right_eye_inner -> right_eye (Approximation)
        7: 3,    # left_ear -> left_ear
        8: 4,    # right_ear -> right_ear
        11: 5,   # left_shoulder -> left_shoulder
        12: 6,   # right_shoulder -> right_shoulder
        13: 7,   # left_elbow -> left_elbow
        14: 8,   # right_elbow -> right_elbow
        15: 9,   # left_wrist -> left_wrist
        16: 10,  # right_wrist -> right_wrist
        23: 11,  # left_hip -> left_hip
        24: 12,  # right_hip -> right_hip
        25: 13,  # left_knee -> left_knee
        26: 14,  # right_knee -> right_knee
        27: 15,  # left_ankle -> left_ankle
        28: 16,  # right_ankle -> right_ankle
    }

    # Model URLs
    MODEL_URLS = {
        0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    }

    def __init__(self, model_complexity: int = 2, min_detection_confidence: float = 0.5):
        """
        Initialisiert MediaPipe Pose.

        Args:
            model_complexity: 0 (lite), 1 (full), oder 2 (heavy)
            min_detection_confidence: Minimale Confidence für Detection
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self._landmarker = None
        self._model_path = None

    def _download_model(self) -> Path:
        """Lädt das Modell herunter falls nötig."""
        import urllib.request

        model_names = {0: "lite", 1: "full", 2: "heavy"}
        model_dir = Path(__file__).parent.parent.parent.parent / "models"
        model_dir.mkdir(exist_ok=True)

        model_file = model_dir / f"pose_landmarker_{model_names[self.model_complexity]}.task"

        if not model_file.exists():
            print(f"Downloading MediaPipe model ({model_names[self.model_complexity]})...")
            url = self.MODEL_URLS[self.model_complexity]
            urllib.request.urlretrieve(url, model_file)
            print(f"Model saved to {model_file}")

        return model_file

    def _load_model(self):
        """Lazy Loading des Modells."""
        if self._landmarker is None:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = self._download_model()

            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def predict(self, frame: np.ndarray) -> list[Keypoint]:
        """
        Führt Inference auf einem Frame aus.

        Args:
            frame: BGR Image als numpy array (H, W, 3)

        Returns:
            Liste von 17 Keypoints im COCO Format
        """
        import cv2
        import mediapipe as mp

        self._load_model()

        # MediaPipe erwartet RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect pose
        results = self._landmarker.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            # Keine Person erkannt - leere Keypoints zurückgeben
            return [
                Keypoint(x=0.0, y=0.0, confidence=0.0, name=name)
                for name in self.COCO_KEYPOINTS
            ]

        # Erste Person nehmen
        landmarks = results.pose_landmarks[0]

        # Native Keypoints extrahieren
        native_keypoints = []
        for idx, landmark in enumerate(landmarks):
            native_keypoints.append(Keypoint(
                x=landmark.x * w,  # Normalisiert -> Pixel
                y=landmark.y * h,
                confidence=landmark.visibility if hasattr(landmark, 'visibility') else 1.0,
                name=f"mp_{idx}"
            ))

        return self.map_to_coco(native_keypoints)

    def get_model_name(self) -> str:
        model_names = {0: "lite", 1: "full", 2: "heavy"}
        return f"MediaPipe ({model_names[self.model_complexity]})"

    @property
    def native_keypoint_mapping(self) -> dict[int, int]:
        return self.MEDIAPIPE_TO_COCO
