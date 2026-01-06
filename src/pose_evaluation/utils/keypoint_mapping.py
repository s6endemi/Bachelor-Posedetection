"""
Keypoint Mapping Utilities.
"""

# Standard COCO 17 Keypoint Namen
COCO_KEYPOINTS = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]


def get_keypoint_index(name: str) -> int:
    """
    Gibt den Index eines Keypoints anhand des Namens zurück.

    Args:
        name: Keypoint Name (z.B. "left_shoulder")

    Returns:
        Index im COCO Format

    Raises:
        ValueError: Wenn Name nicht gefunden
    """
    name_lower = name.lower().replace(" ", "_")

    if name_lower in COCO_KEYPOINTS:
        return COCO_KEYPOINTS.index(name_lower)

    raise ValueError(f"Unbekannter Keypoint: {name}")


# Ground Truth Mapping - wird nach Dataset-Analyse ausgefüllt
GT_TO_COCO_MAPPING: dict[str, int] = {
    # TODO: Nach Analyse von joints_names.txt ausfüllen
    # Beispiel:
    # "Spine": None,  # Kein COCO Equivalent
    # "LeftShoulder": 5,
    # "RightShoulder": 6,
}
