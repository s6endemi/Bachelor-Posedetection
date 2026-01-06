"""
Metriken für Pose Estimation Evaluation.
"""

import numpy as np
from typing import Optional


def calculate_euclidean_error(
    gt_keypoints: np.ndarray,
    pred_keypoints: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Berechnet den euklidischen Fehler pro Keypoint.

    Args:
        gt_keypoints: Ground Truth (N, 2) - x, y Koordinaten
        pred_keypoints: Predictions (N, 2) - x, y Koordinaten
        mask: Optional (N,) - True für gültige Keypoints

    Returns:
        Fehler pro Keypoint (N,)
    """
    errors = np.sqrt(np.sum((gt_keypoints - pred_keypoints) ** 2, axis=1))

    if mask is not None:
        errors = np.where(mask, errors, np.nan)

    return errors


def calculate_torso_length(keypoints: np.ndarray) -> float:
    """
    Berechnet die Torso-Länge als Durchschnitt von Schulter-Hüfte Abständen.

    Args:
        keypoints: (17, 2) COCO Keypoints

    Returns:
        Torso-Länge in Pixeln
    """
    # COCO Indizes: left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]

    # Mittelpunkte berechnen
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2

    torso_length = np.sqrt(np.sum((shoulder_center - hip_center) ** 2))

    return torso_length


def calculate_nmpjpe(
    gt_keypoints: np.ndarray,
    pred_keypoints: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Berechnet den Normalized Mean Per Joint Position Error.

    Normalisiert durch Torso-Länge für Vergleichbarkeit.

    Args:
        gt_keypoints: Ground Truth (N, 2)
        pred_keypoints: Predictions (N, 2)
        mask: Optional (N,) - True für gültige Keypoints

    Returns:
        NMPJPE (normalisiert, in %)
    """
    torso_length = calculate_torso_length(gt_keypoints)

    if torso_length < 1e-6:  # Vermeidung von Division durch 0
        return np.nan

    errors = calculate_euclidean_error(gt_keypoints, pred_keypoints, mask)

    # Normalisieren und Mittelwert
    normalized_errors = errors / torso_length
    nmpjpe = np.nanmean(normalized_errors) * 100  # In Prozent

    return nmpjpe


def calculate_pck(
    gt_keypoints: np.ndarray,
    pred_keypoints: np.ndarray,
    threshold: float = 0.1,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Berechnet Percentage of Correct Keypoints (PCK).

    Args:
        gt_keypoints: Ground Truth (N, 2)
        pred_keypoints: Predictions (N, 2)
        threshold: Threshold als Anteil der Torso-Länge (default: 0.1 = 10%)
        mask: Optional (N,) - True für gültige Keypoints

    Returns:
        PCK in % (0-100)
    """
    torso_length = calculate_torso_length(gt_keypoints)

    if torso_length < 1e-6:
        return np.nan

    errors = calculate_euclidean_error(gt_keypoints, pred_keypoints, mask)
    threshold_pixels = threshold * torso_length

    correct = errors < threshold_pixels

    if mask is not None:
        correct = correct & mask
        num_valid = np.sum(mask)
    else:
        num_valid = len(errors)

    if num_valid == 0:
        return np.nan

    pck = np.sum(correct) / num_valid * 100

    return pck
