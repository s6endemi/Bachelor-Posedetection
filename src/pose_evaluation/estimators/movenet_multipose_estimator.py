"""
MoveNet MultiPose Estimator Implementation.

MultiPose Variante kann bis zu 6 Personen detektieren.
Wird verwendet nachdem SinglePose Limitation identifiziert wurde.
"""

import numpy as np
from .base import PoseEstimator, Keypoint


class MoveNetMultiPoseEstimator(PoseEstimator):
    """MoveNet MultiPose Implementation (Bottom-Up, Multi-Person)."""

    # MoveNet verwendet direkt COCO Keypoints (17)
    MOVENET_TO_COCO = {i: i for i in range(17)}

    def __init__(self):
        """
        Initialisiert MoveNet MultiPose.

        Nur Lightning verfuegbar (schneller, fuer Multi-Person optimiert).
        """
        self._model = None
        self._input_size = 256

    def _load_model(self):
        """Lazy Loading des Modells von TensorFlow Hub."""
        if self._model is None:
            import tensorflow_hub as hub

            url = "https://tfhub.dev/google/movenet/multipose/lightning/1"
            self._model = hub.load(url)
            self._movenet = self._model.signatures["serving_default"]

    def predict(self, frame: np.ndarray) -> list[Keypoint]:
        """
        Fuehrt Inference auf einem Frame aus.

        Args:
            frame: BGR Image als numpy array (H, W, 3)

        Returns:
            Liste von 17 Keypoints im COCO Format (groesste Person)
        """
        import cv2
        import tensorflow as tf

        self._load_model()

        h, w = frame.shape[:2]

        # Preprocessing: Resize und RGB konvertieren
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MultiPose erwartet variable Groesse, aber optimal bei 256x256
        # Behalte Aspect Ratio und padde
        target_size = self._input_size
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb_frame, (new_w, new_h))

        # Padding zu quadratischem Bild
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        pad_y = (target_size - new_h) // 2
        pad_x = (target_size - new_w) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        # TensorFlow Input vorbereiten
        input_tensor = tf.cast(padded, dtype=tf.int32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)

        # Inference
        outputs = self._movenet(input_tensor)
        # MultiPose Output: (1, 6, 56) - 6 Personen, 17*3 + 5 (bbox + score)
        keypoints_with_scores = outputs["output_0"].numpy()[0]  # Shape: (6, 56)

        # Finde die Person mit der groessten Bounding Box
        best_idx = 0
        best_area = 0

        for i in range(6):
            person = keypoints_with_scores[i]
            # Letzte 5 Werte: [ymin, xmin, ymax, xmax, score]
            bbox = person[51:55]
            ymin, xmin, ymax, xmax = bbox
            area = (xmax - xmin) * (ymax - ymin)
            score = person[55]

            # Nur Personen mit Score > 0.1 beruecksichtigen
            if score > 0.1 and area > best_area:
                best_area = area
                best_idx = i

        # Wenn keine Person gefunden
        if best_area == 0:
            return [
                Keypoint(x=0.0, y=0.0, confidence=0.0, name=name)
                for name in self.COCO_KEYPOINTS
            ]

        # Beste Person extrahieren
        best_person = keypoints_with_scores[best_idx]

        # Keypoints extrahieren (erste 51 Werte = 17 * 3)
        native_keypoints = []
        for idx in range(17):
            y_norm = best_person[idx * 3]
            x_norm = best_person[idx * 3 + 1]
            confidence = best_person[idx * 3 + 2]

            # Koordinaten zurueck auf Original-Bildgroesse umrechnen
            # Beruecksichtige Padding und Skalierung
            x_padded = x_norm * target_size
            y_padded = y_norm * target_size

            # Padding entfernen
            x_unpadded = x_padded - pad_x
            y_unpadded = y_padded - pad_y

            # Skalierung rueckgaengig
            x_orig = x_unpadded / scale
            y_orig = y_unpadded / scale

            native_keypoints.append(Keypoint(
                x=float(x_orig),
                y=float(y_orig),
                confidence=float(confidence),
                name=self.COCO_KEYPOINTS[idx]
            ))

        return self.map_to_coco(native_keypoints)

    def get_model_name(self) -> str:
        return "MoveNet (multipose)"

    @property
    def native_keypoint_mapping(self) -> dict[int, int]:
        return self.MOVENET_TO_COCO
