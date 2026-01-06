"""
Abstrakte Basisklasse für Pose Estimation Modelle.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class Keypoint:
    """Einzelner Keypoint mit Koordinaten und Confidence."""
    x: float
    y: float
    confidence: float
    name: str


@dataclass
class PoseResult:
    """Ergebnis einer Pose Estimation."""
    keypoints: list[Keypoint]
    model_name: str
    frame_idx: int


class PoseEstimator(ABC):
    """Abstrakte Basisklasse für alle Pose Estimation Modelle."""

    # COCO Keypoint Namen (17 Keypoints)
    COCO_KEYPOINTS = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Keypoint]:
        """
        Führt Inference auf einem Frame aus.

        Args:
            frame: BGR Image als numpy array (H, W, 3)

        Returns:
            Liste von 17 Keypoints im COCO Format
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Gibt den Modellnamen zurück."""
        pass

    @property
    @abstractmethod
    def native_keypoint_mapping(self) -> dict[int, int]:
        """
        Mapping von nativem Model-Index zu COCO Index.

        Returns:
            Dict mit {model_idx: coco_idx}
        """
        pass

    def map_to_coco(self, native_keypoints: list[Keypoint]) -> list[Keypoint]:
        """
        Mappt native Keypoints auf COCO Format.

        Args:
            native_keypoints: Keypoints im nativen Format des Modells

        Returns:
            17 Keypoints im COCO Format (fehlende mit confidence=0)
        """
        coco_keypoints = [
            Keypoint(x=0.0, y=0.0, confidence=0.0, name=name)
            for name in self.COCO_KEYPOINTS
        ]

        for native_idx, coco_idx in self.native_keypoint_mapping.items():
            if native_idx < len(native_keypoints):
                kp = native_keypoints[native_idx]
                coco_keypoints[coco_idx] = Keypoint(
                    x=kp.x,
                    y=kp.y,
                    confidence=kp.confidence,
                    name=self.COCO_KEYPOINTS[coco_idx]
                )

        return coco_keypoints
