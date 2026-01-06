"""
Evaluation Pipeline für Pose Estimation Vergleich.
"""

import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm

from ..estimators.base import PoseEstimator, Keypoint
from .metrics import calculate_nmpjpe, calculate_pck, calculate_euclidean_error


@dataclass
class FrameResult:
    """Ergebnis für einen einzelnen Frame."""
    frame_idx: int
    rotation_angle: float
    model_name: str
    nmpjpe: float
    pck: float
    per_joint_errors: np.ndarray


@dataclass
class EvaluationResult:
    """Gesamtergebnis der Evaluation."""
    frame_results: list[FrameResult]
    model_name: str

    def to_dataframe(self):
        """Konvertiert zu Pandas DataFrame."""
        import pandas as pd

        records = []
        for fr in self.frame_results:
            record = {
                "frame_idx": fr.frame_idx,
                "rotation_angle": fr.rotation_angle,
                "model_name": fr.model_name,
                "nmpjpe": fr.nmpjpe,
                "pck": fr.pck,
            }
            # Per-Joint Errors hinzufügen
            for i, error in enumerate(fr.per_joint_errors):
                record[f"error_joint_{i}"] = error
            records.append(record)

        return pd.DataFrame(records)


class EvaluationPipeline:
    """Hauptklasse für die Evaluation aller Modelle."""

    def __init__(self, estimators: list[PoseEstimator]):
        """
        Initialisiert die Pipeline.

        Args:
            estimators: Liste von PoseEstimator Instanzen
        """
        self.estimators = estimators

    def run_single_video(
        self,
        video_path: str,
        gt_2d: np.ndarray,
        rotation_angles: np.ndarray,
        progress_bar: bool = True
    ) -> dict[str, EvaluationResult]:
        """
        Führt Evaluation für ein einzelnes Video durch.

        Args:
            video_path: Pfad zum Video
            gt_2d: Ground Truth 2D Keypoints (num_frames, 17, 2)
            rotation_angles: Rotationswinkel pro Frame (num_frames,)
            progress_bar: Zeige Fortschrittsbalken

        Returns:
            Dict mit Modellname -> EvaluationResult
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Kann Video nicht öffnen: {video_path}")

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ergebnisse pro Modell initialisieren
        results = {est.get_model_name(): [] for est in self.estimators}

        frame_iter = range(num_frames)
        if progress_bar:
            frame_iter = tqdm(frame_iter, desc="Processing frames")

        for frame_idx in frame_iter:
            ret, frame = cap.read()
            if not ret:
                break

            gt_kp = gt_2d[frame_idx]
            angle = rotation_angles[frame_idx]

            # Inference mit allen Modellen
            for estimator in self.estimators:
                pred_keypoints = estimator.predict(frame)

                # Zu numpy array konvertieren
                pred_array = np.array([[kp.x, kp.y] for kp in pred_keypoints])

                # Metriken berechnen
                nmpjpe = calculate_nmpjpe(gt_kp, pred_array)
                pck = calculate_pck(gt_kp, pred_array, threshold=0.1)
                per_joint_errors = calculate_euclidean_error(gt_kp, pred_array)

                frame_result = FrameResult(
                    frame_idx=frame_idx,
                    rotation_angle=angle,
                    model_name=estimator.get_model_name(),
                    nmpjpe=nmpjpe,
                    pck=pck,
                    per_joint_errors=per_joint_errors
                )

                results[estimator.get_model_name()].append(frame_result)

        cap.release()

        # In EvaluationResult wrappen
        return {
            name: EvaluationResult(frame_results=frames, model_name=name)
            for name, frames in results.items()
        }

    def aggregate_by_angle_bin(
        self,
        results: dict[str, EvaluationResult],
        bin_size: float = 5.0
    ):
        """
        Aggregiert Ergebnisse nach Winkel-Bins.

        Args:
            results: Ergebnisse von run_single_video
            bin_size: Größe der Winkel-Bins in Grad

        Returns:
            Pandas DataFrame mit aggregierten Ergebnissen
        """
        import pandas as pd

        all_dfs = []
        for model_name, eval_result in results.items():
            df = eval_result.to_dataframe()
            all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Winkel-Bins erstellen
        combined_df["angle_bin"] = (combined_df["rotation_angle"] // bin_size) * bin_size

        # Aggregation
        agg_df = combined_df.groupby(["model_name", "angle_bin"]).agg({
            "nmpjpe": ["mean", "std", "count"],
            "pck": ["mean", "std"],
        }).reset_index()

        # Flatten column names
        agg_df.columns = ["_".join(col).strip("_") for col in agg_df.columns]

        return agg_df
