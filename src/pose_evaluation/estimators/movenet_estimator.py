"""
MoveNet Pose Estimator Implementation.
"""

import numpy as np
from .base import PoseEstimator, Keypoint


class MoveNetEstimator(PoseEstimator):
    """MoveNet Thunder Implementation (Bottom-Up Architektur)."""

    # MoveNet verwendet direkt COCO Keypoints (17)
    # Mapping ist 1:1
    MOVENET_TO_COCO = {i: i for i in range(17)}

    def __init__(self, model_name: str = "thunder"):
        """
        Initialisiert MoveNet.

        Args:
            model_name: "thunder" (genauer) oder "lightning" (schneller)
        """
        if model_name not in ["thunder", "lightning"]:
            raise ValueError("model_name muss 'thunder' oder 'lightning' sein")

        self.model_name_variant = model_name
        self._model = None
        self._input_size = 256 if model_name == "thunder" else 192

    def _load_model(self):
        """Lazy Loading des Modells von TensorFlow Hub."""
        if self._model is None:
            import tensorflow_hub as hub

            if self.model_name_variant == "thunder":
                url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
            else:
                url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

            self._model = hub.load(url)
            self._movenet = self._model.signatures["serving_default"]

    def predict(self, frame: np.ndarray) -> list[Keypoint]:
        """
        Führt Inference auf einem Frame aus.

        Args:
            frame: BGR Image als numpy array (H, W, 3)

        Returns:
            Liste von 17 Keypoints im COCO Format
        """
        import cv2
        import tensorflow as tf

        self._load_model()

        h, w = frame.shape[:2]

        # Preprocessing: Resize und RGB konvertieren
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (self._input_size, self._input_size))

        # TensorFlow Input vorbereiten
        input_tensor = tf.cast(resized, dtype=tf.int32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)

        # Inference
        outputs = self._movenet(input_tensor)
        keypoints_with_scores = outputs["output_0"].numpy()[0, 0]  # Shape: (17, 3)

        # Native Keypoints extrahieren
        native_keypoints = []
        for idx, kp in enumerate(keypoints_with_scores):
            y_norm, x_norm, confidence = kp
            native_keypoints.append(Keypoint(
                x=x_norm * w,  # Normalisiert -> Pixel
                y=y_norm * h,
                confidence=float(confidence),
                name=self.COCO_KEYPOINTS[idx]
            ))

        return self.map_to_coco(native_keypoints)

    def get_model_name(self) -> str:
        return f"MoveNet ({self.model_name_variant})"

    @property
    def native_keypoint_mapping(self) -> dict[int, int]:
        return self.MOVENET_TO_COCO
