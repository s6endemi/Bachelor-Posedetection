"""
Keypoint Mapping zwischen Ground Truth (REHAB24-6) und COCO Format.

Das REHAB24-6 Dataset verwendet 26 Skeleton-Joints aus Motion Capture.
COCO (und die meisten Pose Estimators) verwenden 17 Keypoints.

Nur 12 Keypoints sind direkt vergleichbar (keine Gesichts-Keypoints im GT).
"""

from dataclasses import dataclass
import numpy as np


# Ground Truth Joint Namen (26 Joints, Motion Capture Hierarchie)
GT_JOINT_NAMES = [
    "Hips", "Spine", "Spine1", "Neck", "Head", "Head_end",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHand_end",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHand_end",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeBase_end",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToeBase_end"
]

# COCO Keypoint Namen (17 Keypoints)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


# Mapping: GT Index -> COCO Index
# Nur 12 von 17 COCO Keypoints haben ein GT-Gegenstück
# (Gesichts-Keypoints 0-4 fehlen im GT)
GT_TO_COCO_MAPPING = {
    7: 5,    # LeftArm -> left_shoulder
    12: 6,   # RightArm -> right_shoulder
    8: 7,    # LeftForeArm -> left_elbow
    13: 8,   # RightForeArm -> right_elbow
    9: 9,    # LeftHand -> left_wrist
    14: 10,  # RightHand -> right_wrist
    16: 11,  # LeftUpLeg -> left_hip
    21: 12,  # RightUpLeg -> right_hip
    17: 13,  # LeftLeg -> left_knee
    22: 14,  # RightLeg -> right_knee
    18: 15,  # LeftFoot -> left_ankle
    23: 16,  # RightFoot -> right_ankle
}

# Inverse Mapping: COCO Index -> GT Index
COCO_TO_GT_MAPPING = {v: k for k, v in GT_TO_COCO_MAPPING.items()}

# Liste der vergleichbaren COCO Keypoint Indices
COMPARABLE_COCO_INDICES = sorted(GT_TO_COCO_MAPPING.values())

# Liste der vergleichbaren GT Joint Indices
COMPARABLE_GT_INDICES = sorted(GT_TO_COCO_MAPPING.keys())


def extract_comparable_gt_keypoints(gt_2d: np.ndarray) -> np.ndarray:
    """
    Extrahiert die 12 vergleichbaren Keypoints aus GT-Daten.

    Args:
        gt_2d: Ground Truth 2D Keypoints, Shape (num_frames, 26, 2)

    Returns:
        Gefilterte Keypoints, Shape (num_frames, 12, 2) in COCO-Reihenfolge
    """
    num_frames = gt_2d.shape[0]
    result = np.zeros((num_frames, 12, 2), dtype=gt_2d.dtype)

    for i, coco_idx in enumerate(COMPARABLE_COCO_INDICES):
        gt_idx = COCO_TO_GT_MAPPING[coco_idx]
        result[:, i, :] = gt_2d[:, gt_idx, :]

    return result


def extract_comparable_pred_keypoints(predictions: np.ndarray) -> np.ndarray:
    """
    Extrahiert die 12 vergleichbaren Keypoints aus Predictions.

    Args:
        predictions: Predicted Keypoints im COCO Format, Shape (num_frames, 17, 2/3)

    Returns:
        Gefilterte Keypoints, Shape (num_frames, 12, 2/3)
    """
    num_frames = predictions.shape[0]
    num_coords = predictions.shape[2]  # 2 (x,y) oder 3 (x,y,conf)
    result = np.zeros((num_frames, 12, num_coords), dtype=predictions.dtype)

    for i, coco_idx in enumerate(COMPARABLE_COCO_INDICES):
        result[:, i, :] = predictions[:, coco_idx, :]

    return result


def get_comparable_keypoint_names() -> list[str]:
    """Gibt die Namen der 12 vergleichbaren Keypoints zurück (COCO Namen)."""
    return [COCO_KEYPOINT_NAMES[i] for i in COMPARABLE_COCO_INDICES]


@dataclass
class KeypointPair:
    """Ein Paar aus GT und COCO Keypoint Info."""
    gt_index: int
    gt_name: str
    coco_index: int
    coco_name: str


def get_mapping_info() -> list[KeypointPair]:
    """Gibt detaillierte Mapping-Informationen zurück."""
    pairs = []
    for gt_idx, coco_idx in GT_TO_COCO_MAPPING.items():
        pairs.append(KeypointPair(
            gt_index=gt_idx,
            gt_name=GT_JOINT_NAMES[gt_idx],
            coco_index=coco_idx,
            coco_name=COCO_KEYPOINT_NAMES[coco_idx]
        ))
    return sorted(pairs, key=lambda p: p.coco_index)


if __name__ == "__main__":
    # Mapping ausgeben
    print("GT -> COCO Keypoint Mapping")
    print("=" * 50)
    for pair in get_mapping_info():
        print(f"GT[{pair.gt_index:2d}] {pair.gt_name:20s} -> "
              f"COCO[{pair.coco_index:2d}] {pair.coco_name}")

    print(f"\n{len(GT_TO_COCO_MAPPING)} von 17 COCO Keypoints vergleichbar")
    print("Fehlend: nose, left_eye, right_eye, left_ear, right_ear")
