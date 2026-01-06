"""
Rotationswinkel Berechnung aus 3D Ground Truth.
"""

import numpy as np


def calculate_rotation_angle(
    left_shoulder_3d: np.ndarray,
    right_shoulder_3d: np.ndarray
) -> float:
    """
    Berechnet den Rotationswinkel der Person zur Kamera.

    Basiert auf der Schulterachse in 3D:
    - 0° = Person steht frontal (beide Schultern gleich weit von Kamera)
    - 90° = Person steht seitlich

    Args:
        left_shoulder_3d: (x, y, z) der linken Schulter
        right_shoulder_3d: (x, y, z) der rechten Schulter

    Returns:
        Rotationswinkel in Grad (0-90)
    """
    # Differenz berechnen
    delta_x = right_shoulder_3d[0] - left_shoulder_3d[0]
    delta_z = right_shoulder_3d[2] - left_shoulder_3d[2]

    # Winkel berechnen
    angle_rad = np.arctan2(np.abs(delta_z), np.abs(delta_x))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_rotation_angles_batch(
    keypoints_3d: np.ndarray,
    left_shoulder_idx: int,
    right_shoulder_idx: int
) -> np.ndarray:
    """
    Berechnet Rotationswinkel für mehrere Frames.

    Args:
        keypoints_3d: (num_frames, num_joints, 3)
        left_shoulder_idx: Index der linken Schulter im GT
        right_shoulder_idx: Index der rechten Schulter im GT

    Returns:
        Array mit Winkeln (num_frames,)
    """
    left_shoulders = keypoints_3d[:, left_shoulder_idx, :]
    right_shoulders = keypoints_3d[:, right_shoulder_idx, :]

    angles = np.array([
        calculate_rotation_angle(left, right)
        for left, right in zip(left_shoulders, right_shoulders)
    ])

    return angles
