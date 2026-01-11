"""
Saubere Neu-Evaluation aller Predictions.

Berechnet NMPJPE fuer alle 126 Videos, alle 3 Modelle.
Beruecksichtigt frame_step=3 korrekt.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Korrektes Mapping aus dem Projekt
from src.pose_evaluation.data.keypoint_mapping import (
    COCO_TO_GT_MAPPING,
    COMPARABLE_COCO_INDICES,
    get_comparable_keypoint_names
)


@dataclass
class VideoResult:
    """Ergebnis fuer ein Video."""
    exercise: str
    subject_id: str
    camera: str
    model: str
    num_frames: int
    mean_nmpjpe: float
    std_nmpjpe: float
    median_nmpjpe: float
    frames_over_30: int
    frames_over_50: int
    skip_rate: float  # Anteil Frames mit low confidence


def calculate_frame_nmpjpe(
    pred: np.ndarray,
    gt: np.ndarray,
    min_confidence: float = 0.5
) -> tuple[float, float]:
    """
    Berechnet NMPJPE fuer einen Frame.

    Args:
        pred: (17, 3) Predictions (x, y, confidence)
        gt: (26, 2) Ground Truth
        min_confidence: Minimum confidence fuer Joint-Inclusion

    Returns:
        (nmpjpe, skip_rate) - skip_rate = Anteil gefilterter Joints
    """
    # Vergleichbare Keypoints extrahieren
    pred_pts = []
    gt_pts = []
    confidences = []

    for coco_idx in COMPARABLE_COCO_INDICES:
        gt_idx = COCO_TO_GT_MAPPING[coco_idx]
        pred_pts.append(pred[coco_idx, :2])
        gt_pts.append(gt[gt_idx])
        confidences.append(pred[coco_idx, 2])

    pred_pts = np.array(pred_pts)
    gt_pts = np.array(gt_pts)
    confidences = np.array(confidences)

    # Torso-Laenge berechnen (GT Indices: 7=L_Shoulder, 12=R_Shoulder, 16=L_Hip, 21=R_Hip)
    shoulder_mid = (gt[7] + gt[12]) / 2
    hip_mid = (gt[16] + gt[21]) / 2
    torso_length = np.linalg.norm(shoulder_mid - hip_mid)

    if torso_length < 10:  # Zu klein, ungueltig
        return np.nan, 1.0

    # Fehler berechnen
    errors = np.linalg.norm(pred_pts - gt_pts, axis=1)
    normalized_errors = errors / torso_length * 100

    # Confidence-Filter
    valid_mask = confidences >= min_confidence
    skip_rate = 1.0 - (np.sum(valid_mask) / len(valid_mask))

    if np.sum(valid_mask) == 0:
        return np.nan, skip_rate

    # NMPJPE = Mittelwert der validen normalisierten Fehler
    nmpjpe = np.mean(normalized_errors[valid_mask])

    return nmpjpe, skip_rate


def evaluate_video(
    npz_path: Path,
    gt_2d_dir: Path,
    frame_step: int = 3
) -> list[VideoResult]:
    """
    Evaluiert ein einzelnes Video fuer alle Modelle.

    Args:
        npz_path: Pfad zur .npz Prediction-Datei
        gt_2d_dir: Pfad zum GT 2D Ordner
        frame_step: Frame-Stepping das bei Inference verwendet wurde

    Returns:
        Liste von VideoResult (eines pro Modell)
    """
    # Metadaten aus Pfad
    exercise = npz_path.parent.name
    filename = npz_path.stem
    parts = filename.split("-")
    subject_id = parts[0]
    camera = parts[1]

    # Predictions laden
    data = np.load(npz_path)
    num_frames = int(data['num_frames'])

    # GT laden
    gt_path = gt_2d_dir / exercise / f"{subject_id}-{camera}-30fps.npy"
    if not gt_path.exists():
        print(f"  [WARN] GT nicht gefunden: {gt_path}")
        return []
    gt_2d = np.load(gt_path)

    # Modelle finden
    model_keys = [k for k in data.files if k.startswith('pred_')]

    results = []

    for model_key in model_keys:
        predictions = data[model_key]
        model_name = model_key.replace('pred_', '').replace('_', ' ')

        nmpjpes = []
        skip_rates = []

        for i in range(len(predictions)):
            # WICHTIG: GT-Index beruecksichtigt frame_step
            gt_idx = i * frame_step

            if gt_idx >= len(gt_2d):
                break

            nmpjpe, skip_rate = calculate_frame_nmpjpe(
                predictions[i],
                gt_2d[gt_idx]
            )

            if not np.isnan(nmpjpe):
                nmpjpes.append(nmpjpe)
                skip_rates.append(skip_rate)

        if len(nmpjpes) == 0:
            continue

        nmpjpes = np.array(nmpjpes)

        results.append(VideoResult(
            exercise=exercise,
            subject_id=subject_id,
            camera=camera,
            model=model_name,
            num_frames=len(nmpjpes),
            mean_nmpjpe=np.mean(nmpjpes),
            std_nmpjpe=np.std(nmpjpes),
            median_nmpjpe=np.median(nmpjpes),
            frames_over_30=np.sum(nmpjpes > 30),
            frames_over_50=np.sum(nmpjpes > 50),
            skip_rate=np.mean(skip_rates)
        ))

    return results


def evaluate_all(
    predictions_dir: Path,
    gt_2d_dir: Path,
    frame_step: int = 3
) -> pd.DataFrame:
    """
    Evaluiert alle Predictions.

    Returns:
        DataFrame mit allen Video-Ergebnissen
    """
    npz_files = sorted(predictions_dir.rglob("*.npz"))
    print(f"Gefunden: {len(npz_files)} Prediction-Dateien")

    all_results = []

    for i, npz_path in enumerate(npz_files):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(npz_files)}] {npz_path.parent.name}/{npz_path.name}")

        results = evaluate_video(npz_path, gt_2d_dir, frame_step)
        all_results.extend(results)

    # In DataFrame konvertieren
    df = pd.DataFrame([
        {
            'exercise': r.exercise,
            'subject_id': r.subject_id,
            'camera': r.camera,
            'model': r.model,
            'num_frames': r.num_frames,
            'mean_nmpjpe': r.mean_nmpjpe,
            'std_nmpjpe': r.std_nmpjpe,
            'median_nmpjpe': r.median_nmpjpe,
            'frames_over_30': r.frames_over_30,
            'frames_over_50': r.frames_over_50,
            'skip_rate': r.skip_rate
        }
        for r in all_results
    ])

    return df


def print_summary(df: pd.DataFrame):
    """Gibt eine Zusammenfassung aus."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Gesamt-Statistik pro Modell
    print("\n### Gesamt (alle Videos) ###")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        total_frames = model_df['num_frames'].sum()
        mean_nmpjpe = model_df['mean_nmpjpe'].mean()
        std = model_df['mean_nmpjpe'].std()
        print(f"{model:30s}: {mean_nmpjpe:5.1f}% +/- {std:5.1f}% (n={total_frames})")

    # Nach Kamera
    print("\n### Nach Kamera ###")
    for camera in ['c17', 'c18']:
        print(f"\n{camera}:")
        cam_df = df[df['camera'] == camera]
        for model in df['model'].unique():
            model_df = cam_df[cam_df['model'] == model]
            if len(model_df) == 0:
                continue
            mean_nmpjpe = model_df['mean_nmpjpe'].mean()
            videos_over_30 = (model_df['mean_nmpjpe'] > 30).sum()
            total_videos = len(model_df)
            pct_over_30 = videos_over_30 / total_videos * 100
            print(f"  {model:28s}: {mean_nmpjpe:5.1f}% (Videos >30%: {videos_over_30}/{total_videos} = {pct_over_30:.1f}%)")

    # Ausreisser identifizieren
    print("\n### Ausreisser (Videos mit >30% NMPJPE) ###")
    outliers = df[df['mean_nmpjpe'] > 30].sort_values('mean_nmpjpe', ascending=False)
    if len(outliers) > 0:
        for _, row in outliers.head(10).iterrows():
            print(f"  {row['subject_id']}-{row['camera']} ({row['model']}): {row['mean_nmpjpe']:.1f}%")
    else:
        print("  Keine Ausreisser gefunden!")


if __name__ == "__main__":
    predictions_dir = Path("data/predictions")
    gt_2d_dir = Path("data/gt_2d")

    print("Starte saubere Neu-Evaluation...")
    print(f"Predictions: {predictions_dir}")
    print(f"Ground Truth: {gt_2d_dir}")
    print(f"Frame-Step: 3")
    print()

    df = evaluate_all(predictions_dir, gt_2d_dir, frame_step=3)

    # Zusammenfassung ausgeben
    print_summary(df)

    # Speichern
    output_path = Path("data/evaluation_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nErgebnisse gespeichert: {output_path}")
