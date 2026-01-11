"""
YOLOv8-Pose Estimator Implementation.
"""

import numpy as np
from pathlib import Path
from .base import PoseEstimator, Keypoint


class YOLOPoseEstimator(PoseEstimator):
    """YOLOv8-Pose Implementation (One-Stage Architektur)."""

    # YOLOv8-Pose verwendet direkt COCO Keypoints (17)
    # Mapping ist 1:1
    YOLO_TO_COCO = {i: i for i in range(17)}

    def __init__(self, model_size: str = "m"):
        """
        Initialisiert YOLOv8-Pose.

        Args:
            model_size: "n" (nano), "s" (small), "m" (medium), "l" (large), "x" (xlarge)
        """
        valid_sizes = ["n", "s", "m", "l", "x"]
        if model_size not in valid_sizes:
            raise ValueError(f"model_size muss einer von {valid_sizes} sein")

        self.model_size = model_size
        self._model = None

    def _get_model_path(self) -> Path:
        """Gibt den Pfad zum Modell zurück (im models/ Ordner)."""
        model_dir = Path(__file__).parent.parent.parent.parent / "models"
        model_dir.mkdir(exist_ok=True)
        return model_dir / f"yolov8{self.model_size}-pose.pt"

    def _load_model(self):
        """Lazy Loading des Modells."""
        if self._model is None:
            from ultralytics import YOLO
            model_path = self._get_model_path()
            # YOLO lädt automatisch herunter wenn nicht vorhanden
            self._model = YOLO(str(model_path))

    def predict(self, frame: np.ndarray) -> list[Keypoint]:
        """
        Führt Inference auf einem Frame aus.

        Args:
            frame: BGR Image als numpy array (H, W, 3)

        Returns:
            Liste von 17 Keypoints im COCO Format
        """
        self._load_model()

        # YOLO Inference (verbose=False um Output zu unterdrücken)
        results = self._model(frame, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            # Keine Person erkannt
            return [
                Keypoint(x=0.0, y=0.0, confidence=0.0, name=name)
                for name in self.COCO_KEYPOINTS
            ]

        keypoints_data = results[0].keypoints

        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return [
                Keypoint(x=0.0, y=0.0, confidence=0.0, name=name)
                for name in self.COCO_KEYPOINTS
            ]

        # Bei mehreren Personen: waehle die groesste MIT ausreichender Confidence
        # Score-Filter verhindert Auswahl von teilweise sichtbaren Personen (z.B. Coach)
        num_persons = len(keypoints_data.xy)
        if num_persons > 1 and results[0].boxes is not None:
            best_idx = 0
            best_area = 0
            boxes = results[0].boxes
            for i in range(min(num_persons, len(boxes))):
                box = boxes.xyxy[i].cpu().numpy()
                area = (box[2] - box[0]) * (box[3] - box[1])
                # Score-Filter: ignoriere Detections mit niedriger Confidence
                score = float(boxes.conf[i].cpu().numpy()) if boxes.conf is not None else 1.0
                if score > 0.3 and area > best_area:
                    best_area = area
                    best_idx = i
            person_idx = best_idx
        else:
            person_idx = 0

        # Koordinaten und Confidence extrahieren
        xy = keypoints_data.xy[person_idx].cpu().numpy()  # Shape: (17, 2)
        conf = keypoints_data.conf[person_idx].cpu().numpy() if keypoints_data.conf is not None else np.ones(17)

        # Native Keypoints extrahieren
        native_keypoints = []
        for idx in range(17):
            native_keypoints.append(Keypoint(
                x=float(xy[idx, 0]),
                y=float(xy[idx, 1]),
                confidence=float(conf[idx]),
                name=self.COCO_KEYPOINTS[idx]
            ))

        return self.map_to_coco(native_keypoints)

    def get_model_name(self) -> str:
        return f"YOLOv8-Pose ({self.model_size})"

    @property
    def native_keypoint_mapping(self) -> dict[int, int]:
        return self.YOLO_TO_COCO
