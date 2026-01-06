"""
Validiert das Keypoint-Mapping durch visuelle Überprüfung.
Zeigt GT-Keypoints auf einem Video-Frame mit Beschriftung.
"""

import numpy as np
import cv2
from pathlib import Path

# Joint Namen aus dem Dataset
GT_JOINT_NAMES = [
    "Hips", "Spine", "Spine1", "Neck", "Head", "Head_end",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHand_end",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHand_end",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeBase_end",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToeBase_end"
]

# Mein vermutetes Mapping: GT Index → COCO Name
PROPOSED_MAPPING = {
    7: "left_shoulder",    # LeftArm
    12: "right_shoulder",  # RightArm
    8: "left_elbow",       # LeftForeArm
    13: "right_elbow",     # RightForeArm
    9: "left_wrist",       # LeftHand
    14: "right_wrist",     # RightHand
    16: "left_hip",        # LeftUpLeg
    21: "right_hip",       # RightUpLeg
    17: "left_knee",       # LeftLeg
    22: "right_knee",      # RightLeg
    18: "left_ankle",      # LeftFoot
    23: "right_ankle",     # RightFoot
}

def load_data():
    """Lädt ein Video-Frame und die zugehörigen GT-Daten."""
    # Pfade
    video_path = Path("data/videos/Ex1/PM_000-Camera17-30fps.mp4")
    gt_2d_path = Path("data/gt_2d/Ex1/PM_000-c17-30fps.npy")
    gt_3d_path = Path("data/gt_3d/Ex1/PM_000-30fps.npy")

    # Video Frame laden
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Konnte Video nicht laden")

    # GT laden
    gt_2d = np.load(gt_2d_path)
    gt_3d = np.load(gt_3d_path)

    print(f"Video Frame Shape: {frame.shape}")
    print(f"GT 2D Shape: {gt_2d.shape}")
    print(f"GT 3D Shape: {gt_3d.shape}")

    return frame, gt_2d, gt_3d


def visualize_all_joints(frame, gt_2d, frame_idx=0):
    """Visualisiert ALLE GT Joints mit Index und Namen."""
    vis_frame = frame.copy()

    keypoints = gt_2d[frame_idx]  # (26, 2) oder (26, 3)?

    print(f"\nKeypoints für Frame {frame_idx}:")
    print(f"Keypoints Shape: {keypoints.shape}")

    # Farben: Grün für gemappte Joints, Grau für nicht gemappte
    for idx, kp in enumerate(keypoints):
        x, y = int(kp[0]), int(kp[1])

        # Prüfen ob im Bild
        if x < 0 or y < 0 or x > frame.shape[1] or y > frame.shape[0]:
            continue

        # Farbe: Grün wenn im Mapping, sonst Grau
        if idx in PROPOSED_MAPPING:
            color = (0, 255, 0)  # Grün
            label = f"{idx}:{GT_JOINT_NAMES[idx]}→{PROPOSED_MAPPING[idx]}"
        else:
            color = (128, 128, 128)  # Grau
            label = f"{idx}:{GT_JOINT_NAMES[idx]}"

        # Punkt zeichnen
        cv2.circle(vis_frame, (x, y), 5, color, -1)

        # Label zeichnen (mit Hintergrund für Lesbarkeit)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(vis_frame, (x, y - h - 5), (x + w, y), (0, 0, 0), -1)
        cv2.putText(vis_frame, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)

        print(f"  {idx:2d}: {GT_JOINT_NAMES[idx]:20s} -> ({x:4d}, {y:4d})")

    return vis_frame


def visualize_skeleton_connections(frame, gt_2d, frame_idx=0):
    """Zeichnet Skeleton-Verbindungen um die Struktur zu verstehen."""
    vis_frame = frame.copy()
    keypoints = gt_2d[frame_idx]

    # Typische Skeleton-Verbindungen basierend auf der Hierarchie
    connections = [
        # Wirbelsäule
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        # Linker Arm
        (2, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        # Rechter Arm
        (2, 11), (11, 12), (12, 13), (13, 14), (14, 15),
        # Linkes Bein
        (0, 16), (16, 17), (17, 18), (18, 19), (19, 20),
        # Rechtes Bein
        (0, 21), (21, 22), (22, 23), (23, 24), (24, 25),
    ]

    # Verbindungen zeichnen
    for start_idx, end_idx in connections:
        start = keypoints[start_idx]
        end = keypoints[end_idx]

        x1, y1 = int(start[0]), int(start[1])
        x2, y2 = int(end[0]), int(end[1])

        # Prüfen ob gültig
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            continue

        cv2.line(vis_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Punkte zeichnen
    for idx, kp in enumerate(keypoints):
        x, y = int(kp[0]), int(kp[1])
        if x < 0 or y < 0:
            continue

        if idx in PROPOSED_MAPPING:
            color = (0, 255, 0)  # Grün = gemappt
        else:
            color = (0, 0, 255)  # Rot = nicht gemappt

        cv2.circle(vis_frame, (x, y), 8, color, -1)
        cv2.putText(vis_frame, str(idx), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_frame


if __name__ == "__main__":
    print("=" * 60)
    print("KEYPOINT MAPPING VALIDATION")
    print("=" * 60)

    # Daten laden
    frame, gt_2d, gt_3d = load_data()

    # Visualisierung 1: Alle Joints mit Labels
    vis1 = visualize_all_joints(frame, gt_2d, frame_idx=0)
    cv2.imwrite("validation_all_joints.png", vis1)
    print("\n[OK] Gespeichert: validation_all_joints.png")

    # Visualisierung 2: Skeleton mit Verbindungen
    vis2 = visualize_skeleton_connections(frame, gt_2d, frame_idx=0)
    cv2.imwrite("validation_skeleton.png", vis2)
    print("[OK] Gespeichert: validation_skeleton.png")

    print("\n" + "=" * 60)
    print("Open the images to validate the mapping!")
    print("Green = mapped joints, Red/Gray = not mapped")
    print("=" * 60)
