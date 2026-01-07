"""
Evaluator - Berechnet Metriken und aggregiert nach Rotationswinkel.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json

from .metrics import calculate_nmpjpe, calculate_pck, calculate_euclidean_error
from ..data.keypoint_mapping import (
    COCO_TO_GT_MAPPING,
    COMPARABLE_COCO_INDICES,
    get_comparable_keypoint_names
)


@dataclass
class FrameMetrics:
    """Metriken fuer einen einzelnen Frame."""
    frame_idx: int
    rotation_angle: float
    model_name: str
    nmpjpe: float
    pck: float
    per_joint_errors: np.ndarray  # (12,) fuer vergleichbare Joints


@dataclass
class AngleBinStats:
    """Statistiken fuer einen Winkel-Bin."""
    bin_start: float
    bin_end: float
    bin_center: float
    num_frames: int
    nmpjpe_mean: float
    nmpjpe_std: float
    pck_mean: float
    pck_std: float
    per_joint_nmpjpe: dict[str, float]  # Joint name -> mean NMPJPE


class Evaluator:
    """
    Evaluiert Pose Estimation Ergebnisse.

    Laedt Predictions, vergleicht mit GT, aggregiert nach Rotationswinkel.
    """

    # Winkel-Bins: 0-10, 10-20, ..., 80-90
    DEFAULT_ANGLE_BINS = list(range(0, 91, 10))

    # Minimum Confidence fuer Joint-Inclusion
    # Joints mit niedrigerer Confidence werden von der NMPJPE-Berechnung ausgeschlossen
    # Behebt MediaPipe Unterkörper-Ausreißer (siehe docs/02_PROBLEMS_AND_SOLUTIONS.md Problem 7)
    MIN_JOINT_CONFIDENCE = 0.5

    def __init__(
        self,
        predictions_dir: Path,
        gt_2d_dir: Path,
        angle_bins: list[float] = None
    ):
        """
        Args:
            predictions_dir: Ordner mit .npz Prediction-Dateien
            gt_2d_dir: Ordner mit GT 2D .npy Dateien
            angle_bins: Winkel-Bin Grenzen (default: 0, 10, 20, ..., 90)
        """
        self.predictions_dir = Path(predictions_dir)
        self.gt_2d_dir = Path(gt_2d_dir)
        self.angle_bins = angle_bins or self.DEFAULT_ANGLE_BINS

        # Vergleichbare COCO Indices
        self.comparable_indices = COMPARABLE_COCO_INDICES
        self.joint_names = get_comparable_keypoint_names()

    def load_prediction_file(self, npz_path: Path) -> dict:
        """Laedt eine .npz Prediction-Datei."""
        data = np.load(npz_path)
        return {key: data[key] for key in data.files}

    def get_gt_2d_path(self, exercise: str, subject_id: str, camera: str) -> Path:
        """Konstruiert den GT 2D Pfad."""
        return self.gt_2d_dir / exercise / f"{subject_id}-{camera}-30fps.npy"

    def extract_comparable_keypoints_gt(self, gt_2d_frame: np.ndarray) -> np.ndarray:
        """
        Extrahiert die 12 vergleichbaren Keypoints aus GT (26 Joints).

        Args:
            gt_2d_frame: (26, 2) GT Keypoints

        Returns:
            (12, 2) vergleichbare Keypoints in COCO-Reihenfolge
        """
        result = np.zeros((12, 2))
        for i, coco_idx in enumerate(self.comparable_indices):
            gt_idx = COCO_TO_GT_MAPPING[coco_idx]
            result[i] = gt_2d_frame[gt_idx]
        return result

    def extract_comparable_keypoints_pred(self, pred_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Extrahiert die 12 vergleichbaren Keypoints aus Predictions (17 COCO).

        Args:
            pred_frame: (17, 3) Predicted Keypoints (x, y, conf)

        Returns:
            Tuple of:
                - (12, 2) vergleichbare Keypoints (x, y)
                - (12,) Confidence-Werte
        """
        coords = np.zeros((12, 2))
        confidences = np.zeros(12)
        for i, coco_idx in enumerate(self.comparable_indices):
            coords[i] = pred_frame[coco_idx, :2]  # x, y
            confidences[i] = pred_frame[coco_idx, 2]  # confidence
        return coords, confidences

    def calculate_torso_length_gt(self, gt_2d_frame: np.ndarray) -> float:
        """
        Berechnet Torso-Laenge aus GT (26 Joints).

        GT Indices: 7=LeftArm(Schulter), 12=RightArm, 16=LeftUpLeg(Huefte), 21=RightUpLeg
        """
        left_shoulder = gt_2d_frame[7]
        right_shoulder = gt_2d_frame[12]
        left_hip = gt_2d_frame[16]
        right_hip = gt_2d_frame[21]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        return np.sqrt(np.sum((shoulder_center - hip_center) ** 2))

    def evaluate_frame(
        self,
        pred_frame: np.ndarray,
        gt_2d_frame: np.ndarray,
        rotation_angle: float,
        model_name: str,
        frame_idx: int
    ) -> FrameMetrics:
        """
        Evaluiert einen einzelnen Frame.

        Args:
            pred_frame: (17, 3) Predicted Keypoints
            gt_2d_frame: (26, 2) GT Keypoints
            rotation_angle: Rotationswinkel in Grad
            model_name: Name des Modells
            frame_idx: Frame Index

        Returns:
            FrameMetrics
        """
        # Vergleichbare Keypoints extrahieren
        gt_comparable = self.extract_comparable_keypoints_gt(gt_2d_frame)
        pred_comparable, pred_confidences = self.extract_comparable_keypoints_pred(pred_frame)

        # Torso-Laenge aus GT
        torso_length = self.calculate_torso_length_gt(gt_2d_frame)

        if torso_length < 1e-6:
            return FrameMetrics(
                frame_idx=frame_idx,
                rotation_angle=rotation_angle,
                model_name=model_name,
                nmpjpe=np.nan,
                pck=np.nan,
                per_joint_errors=np.full(12, np.nan)
            )

        # Fehler berechnen
        errors = calculate_euclidean_error(gt_comparable, pred_comparable)
        normalized_errors = errors / torso_length * 100  # In Prozent

        # Confidence-Filter anwenden: Joints mit niedriger Confidence ausschliessen
        # Setzt Fehler auf NaN fuer low-confidence Joints, damit sie bei nanmean ignoriert werden
        valid_mask = pred_confidences >= self.MIN_JOINT_CONFIDENCE
        filtered_errors = np.where(valid_mask, normalized_errors, np.nan)

        # NMPJPE = Mittelwert der normalisierten Fehler (nur valide Joints)
        nmpjpe = np.nanmean(filtered_errors)

        # PCK = Anteil der Keypoints unter Threshold (nur valide Joints)
        threshold = 0.1 * torso_length
        valid_errors = errors[valid_mask]
        if len(valid_errors) > 0:
            pck = np.sum(valid_errors < threshold) / len(valid_errors) * 100
        else:
            pck = np.nan

        return FrameMetrics(
            frame_idx=frame_idx,
            rotation_angle=rotation_angle,
            model_name=model_name,
            nmpjpe=nmpjpe,
            pck=pck,
            per_joint_errors=filtered_errors  # NaN fuer gefilterte Joints
        )

    def evaluate_video(
        self,
        npz_path: Path,
        exercise: str,
        subject_id: str,
        camera: str
    ) -> list[FrameMetrics]:
        """
        Evaluiert alle Frames eines Videos.

        Returns:
            Liste von FrameMetrics (pro Frame, pro Modell)
        """
        # Predictions laden
        pred_data = self.load_prediction_file(npz_path)
        rotation_angles = pred_data["rotation_angles"]
        num_frames = int(pred_data["num_frames"])

        # GT laden
        gt_2d_path = self.get_gt_2d_path(exercise, subject_id, camera)
        gt_2d = np.load(gt_2d_path)

        # Modell-Keys finden (pred_MediaPipe_heavy, pred_MoveNet_thunder, etc.)
        model_keys = [k for k in pred_data.keys() if k.startswith("pred_")]

        all_metrics = []

        for frame_idx in range(min(num_frames, len(gt_2d), len(rotation_angles))):
            for model_key in model_keys:
                # Modellname extrahieren (pred_MediaPipe_heavy -> MediaPipe (heavy))
                model_name = model_key.replace("pred_", "").replace("_", " ")

                pred_frame = pred_data[model_key][frame_idx]
                gt_frame = gt_2d[frame_idx]
                rotation = rotation_angles[frame_idx]

                metrics = self.evaluate_frame(
                    pred_frame, gt_frame, rotation, model_name, frame_idx
                )
                all_metrics.append(metrics)

        return all_metrics

    def aggregate_by_angle_bins(
        self,
        all_metrics: list[FrameMetrics]
    ) -> dict[str, list[AngleBinStats]]:
        """
        Aggregiert Metriken nach Modell und Winkel-Bins.

        Returns:
            {model_name: [AngleBinStats, ...]}
        """
        # In DataFrame konvertieren
        data = []
        for m in all_metrics:
            row = {
                "model": m.model_name,
                "rotation": m.rotation_angle,
                "nmpjpe": m.nmpjpe,
                "pck": m.pck,
            }
            # Per-Joint Errors hinzufuegen
            for i, joint_name in enumerate(self.joint_names):
                row[f"error_{joint_name}"] = m.per_joint_errors[i]
            data.append(row)

        df = pd.DataFrame(data)

        # Winkel-Bin zuweisen
        df["angle_bin"] = pd.cut(
            df["rotation"],
            bins=self.angle_bins,
            labels=[f"{self.angle_bins[i]}-{self.angle_bins[i+1]}"
                    for i in range(len(self.angle_bins)-1)],
            include_lowest=True
        )

        # Pro Modell aggregieren
        results = {}

        for model_name in df["model"].unique():
            model_df = df[df["model"] == model_name]
            bin_stats = []

            for i in range(len(self.angle_bins) - 1):
                bin_start = self.angle_bins[i]
                bin_end = self.angle_bins[i + 1]
                bin_label = f"{bin_start}-{bin_end}"

                bin_df = model_df[model_df["angle_bin"] == bin_label]

                if len(bin_df) == 0:
                    continue

                # Per-Joint NMPJPE
                per_joint = {}
                for joint_name in self.joint_names:
                    col = f"error_{joint_name}"
                    per_joint[joint_name] = bin_df[col].mean()

                stats = AngleBinStats(
                    bin_start=bin_start,
                    bin_end=bin_end,
                    bin_center=(bin_start + bin_end) / 2,
                    num_frames=len(bin_df),
                    nmpjpe_mean=bin_df["nmpjpe"].mean(),
                    nmpjpe_std=bin_df["nmpjpe"].std(),
                    pck_mean=bin_df["pck"].mean(),
                    pck_std=bin_df["pck"].std(),
                    per_joint_nmpjpe=per_joint
                )
                bin_stats.append(stats)

            results[model_name] = bin_stats

        return results

    def evaluate_all(self, max_videos: int = None) -> dict[str, list[AngleBinStats]]:
        """
        Evaluiert alle Prediction-Dateien.

        Args:
            max_videos: Optional - nur erste N Videos

        Returns:
            {model_name: [AngleBinStats, ...]}
        """
        all_metrics = []

        # Alle .npz Dateien finden
        npz_files = list(self.predictions_dir.rglob("*.npz"))

        if max_videos:
            npz_files = npz_files[:max_videos]

        print(f"Evaluating {len(npz_files)} prediction files...")

        for i, npz_path in enumerate(npz_files):
            # Metadaten aus Pfad extrahieren
            # predictions/Ex1/PM_000-c17.npz
            exercise = npz_path.parent.name
            filename = npz_path.stem  # PM_000-c17
            parts = filename.split("-")
            subject_id = parts[0]  # PM_000
            camera = parts[1]      # c17

            print(f"  [{i+1}/{len(npz_files)}] {exercise}/{filename}")

            try:
                metrics = self.evaluate_video(npz_path, exercise, subject_id, camera)
                all_metrics.extend(metrics)
            except Exception as e:
                print(f"    Error: {e}")

        print(f"\nTotal frames evaluated: {len(all_metrics)}")

        # Aggregieren
        return self.aggregate_by_angle_bins(all_metrics)

    def results_to_dataframe(
        self,
        results: dict[str, list[AngleBinStats]]
    ) -> pd.DataFrame:
        """Konvertiert Ergebnisse in einen DataFrame."""
        rows = []
        for model_name, bin_stats in results.items():
            for stats in bin_stats:
                row = {
                    "model": model_name,
                    "angle_bin": f"{int(stats.bin_start)}-{int(stats.bin_end)}",
                    "angle_center": stats.bin_center,
                    "num_frames": stats.num_frames,
                    "nmpjpe_mean": stats.nmpjpe_mean,
                    "nmpjpe_std": stats.nmpjpe_std,
                    "pck_mean": stats.pck_mean,
                    "pck_std": stats.pck_std,
                }
                # Per-Joint hinzufuegen
                for joint_name, error in stats.per_joint_nmpjpe.items():
                    row[f"nmpjpe_{joint_name}"] = error
                rows.append(row)

        return pd.DataFrame(rows)

    def save_results(self, results: dict, output_path: Path):
        """Speichert Ergebnisse als CSV und JSON."""
        df = self.results_to_dataframe(results)

        # CSV
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # JSON (fuer einfache Weiterverarbeitung)
        json_path = output_path.with_suffix(".json")
        df.to_json(json_path, orient="records", indent=2)
        print(f"Saved: {json_path}")

        return df


if __name__ == "__main__":
    # Test mit vorhandenen Predictions
    evaluator = Evaluator(
        predictions_dir=Path("data/predictions"),
        gt_2d_dir=Path("data/gt_2d")
    )

    results = evaluator.evaluate_all()

    print("\n=== Ergebnisse ===")
    for model_name, bin_stats in results.items():
        print(f"\n{model_name}:")
        for stats in bin_stats:
            print(f"  {stats.bin_start}-{stats.bin_end} deg: "
                  f"NMPJPE={stats.nmpjpe_mean:.2f}% (+/-{stats.nmpjpe_std:.2f}), "
                  f"PCK={stats.pck_mean:.1f}%, "
                  f"n={stats.num_frames}")
