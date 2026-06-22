"""
Evaluation Pipeline v2 — Central Configuration

All parameters in ONE place. No magic numbers anywhere else.
Every threshold is documented with its rationale.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# =============================================================================
# Keypoint Mapping (shared across all datasets using COCO-17 predictions)
# =============================================================================

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 12 body joints comparable across models and ground truth (no face)
COMPARABLE_COCO_INDICES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

COMPARABLE_JOINT_NAMES = [COCO_KEYPOINT_NAMES[i] for i in COMPARABLE_COCO_INDICES]

# Joint groups for per-region analysis
JOINT_GROUPS = {
    "shoulders": [0, 1],    # indices into COMPARABLE (left_shoulder, right_shoulder)
    "elbows":    [2, 3],
    "wrists":    [4, 5],
    "hips":      [6, 7],
    "knees":     [8, 9],
    "ankles":    [10, 11],
}

# Torso joints for normalization (indices into COMPARABLE_COCO_INDICES)
# left_shoulder=0, right_shoulder=1, left_hip=6, right_hip=7
TORSO_SHOULDER_INDICES = (0, 1)
TORSO_HIP_INDICES = (6, 7)


# =============================================================================
# Evaluation Parameters
# =============================================================================

@dataclass
class EvalConfig:
    """Parameters that control evaluation metrics computation."""

    # --- Confidence filtering ---
    # Joints with confidence below this are excluded from NMPJPE.
    # 0.5 is standard: filters unreliable detections without being too aggressive.
    # MediaPipe: ~7% joints filtered, MoveNet: ~2%, YOLO: ~0%.
    joint_confidence_threshold: float = 0.5

    # --- Outlier definition ---
    # Frames with NMPJPE above this are flagged as outliers (likely person-switch).
    # 100% of torso length = prediction is more than one torso-length away from GT.
    outlier_nmpjpe_threshold: float = 100.0

    # --- Frame alignment ---
    # Must match what was used during inference.
    # frame_step=3 means prediction[i] corresponds to video_frame[i*3].
    frame_step: int = 3

    # --- Rotation angle ---
    # Empirically determined: MoCap angle when person faces camera c17.
    # Based on reference videos PM_114, PM_122, PM_109.
    # Uncertainty: approx ±5°.
    c17_frontal_offset: float = 65.0

    # Bin size for rotation analysis (degrees).
    # 10° absorbs natural variation (±2.5° std from marker movement).
    rotation_bin_size: int = 10

    # --- Multi-person severity ---
    # From Segmentation.csv: extra_person severity >= this counts as multi-person.
    # 0=none, 1=negligible (hand at edge), 2=noticeable (arm visible), 3=large.
    multi_person_severity_threshold: int = 2


# =============================================================================
# Dataset Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""

    name: str
    predictions_dir: Path
    gt_2d_dir: Path
    gt_3d_dir: Path  # for rotation angle computation
    segmentation_csv: Optional[Path]  # multi-person metadata (if available)
    output_dir: Path

    # GT format
    gt_n_joints: int = 26
    gt_fps: int = 30

    # GT-to-COCO keypoint mapping: {gt_index: coco_index}
    gt_to_coco: dict = field(default_factory=dict)

    # GT indices for torso normalization (shoulders and hips)
    gt_shoulder_indices: tuple = (7, 12)   # LeftArm, RightArm
    gt_hip_indices: tuple = (16, 21)       # LeftUpLeg, RightUpLeg

    # GT indices for rotation computation (3D shoulders)
    gt_3d_left_shoulder: int = 7
    gt_3d_right_shoulder: int = 12

    # Model keys in .npz files
    model_keys: list = field(default_factory=list)

    # Segmentation CSV config
    seg_delimiter: str = ";"
    seg_video_col: str = "video_id"
    seg_exercise_col: str = "exercise_id"
    seg_first_frame_col: str = "first_frame"
    seg_last_frame_col: str = "last_frame"
    seg_extra_person_c17_col: str = "extra_person_in_cam17"
    seg_extra_person_c18_col: str = "extra_person_in_cam18"
    seg_orientation_col: str = "cam17_orientation"
    seg_mocap_error_col: str = "mocap_erroneous"


def get_rehab24_config(base_dir: Path) -> DatasetConfig:
    """REHAB24-6 dataset configuration."""
    data_dir = base_dir / "data"
    return DatasetConfig(
        name="REHAB24-6",
        predictions_dir=data_dir / "predictions",
        gt_2d_dir=data_dir / "gt_2d",
        gt_3d_dir=data_dir / "gt_3d",
        segmentation_csv=data_dir / "Segmentation (1).csv",
        output_dir=base_dir / "evaluation_v2" / "results" / "rehab24",
        gt_n_joints=26,
        gt_fps=30,
        gt_to_coco={
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
        },
        gt_shoulder_indices=(7, 12),
        gt_hip_indices=(16, 21),
        gt_3d_left_shoulder=7,
        gt_3d_right_shoulder=12,
        model_keys=[
            "pred_MediaPipe_full",
            "pred_MoveNet_multipose",
            "pred_YOLOv8-Pose_n",
        ],
    )


# Short display names for thesis
MODEL_DISPLAY_NAMES = {
    "pred_MediaPipe_full": "MediaPipe",
    "pred_MoveNet_multipose": "MoveNet",
    "pred_YOLOv8-Pose_n": "YOLOv8",
}
