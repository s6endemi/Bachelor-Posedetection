"""
Investigation: Why does MediaPipe have higher jitter?
Hypothesis: Jitter correlates with detection instability (joints appearing/disappearing)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
df = pd.read_csv(RESULTS_DIR / "frame_level_data.csv")

print("="*70)
print("INVESTIGATION: WHY DOES MEDIAPIPE HAVE HIGHER JITTER?")
print("="*70)

# Filter clean data
df_clean = df[(df['is_coach'] == False) & (df['nmpjpe'] < 100)].copy()

# Hypothesis 1: Jitter correlates with valid_joints changes
print("\n### Hypothesis 1: Detection Instability causes Jitter")
print("If joints appear/disappear between frames, this could cause jitter.\n")

for model in ['MediaPipe', 'MoveNet', 'YOLO']:
    model_data = df_clean[df_clean['model'] == model].copy()

    jitter_with_change = []
    jitter_without_change = []

    for video_id in model_data['video_id'].unique():
        video_data = model_data[model_data['video_id'] == video_id].sort_values('frame_idx')

        if len(video_data) < 2:
            continue

        nmpjpe = video_data['nmpjpe'].values
        valid_joints = video_data['valid_joints'].values

        for i in range(1, len(nmpjpe)):
            jitter = abs(nmpjpe[i] - nmpjpe[i-1])
            joints_changed = valid_joints[i] != valid_joints[i-1]

            if joints_changed:
                jitter_with_change.append(jitter)
            else:
                jitter_without_change.append(jitter)

    mean_with = np.mean(jitter_with_change) if jitter_with_change else 0
    mean_without = np.mean(jitter_without_change) if jitter_without_change else 0
    n_changes = len(jitter_with_change)
    n_stable = len(jitter_without_change)
    pct_changes = 100 * n_changes / (n_changes + n_stable)

    print(f"{model}:")
    print(f"  Frames where valid_joints changed: {n_changes:,} ({pct_changes:.1f}%)")
    print(f"  Mean jitter WHEN joints change:    {mean_with:.2f}%")
    print(f"  Mean jitter WHEN joints stable:    {mean_without:.2f}%")
    print(f"  Ratio: {mean_with/mean_without:.1f}x higher jitter when joints change")
    print()

# Hypothesis 2: Which specific joints cause problems?
print("\n### Hypothesis 2: Which joints are unstable (appear/disappear)?")
print("Counting how often each joint changes detection status between frames.\n")

joint_cols = [
    'error_left_shoulder', 'error_right_shoulder',
    'error_left_elbow', 'error_right_elbow',
    'error_left_wrist', 'error_right_wrist',
    'error_left_hip', 'error_right_hip',
    'error_left_knee', 'error_right_knee',
    'error_left_ankle', 'error_right_ankle'
]

print("| Joint | MediaPipe Flips | MoveNet Flips | YOLO Flips |")
print("|-------|-----------------|---------------|------------|")

for joint in joint_cols:
    joint_name = joint.replace('error_', '').replace('_', ' ').title()
    row = f"| {joint_name:15} |"

    for model in ['MediaPipe', 'MoveNet', 'YOLO']:
        model_data = df_clean[df_clean['model'] == model].copy()

        flip_count = 0
        total_transitions = 0

        for video_id in model_data['video_id'].unique():
            video_data = model_data[model_data['video_id'] == video_id].sort_values('frame_idx')

            if len(video_data) < 2:
                continue

            joint_detected = video_data[joint].notna().values

            for i in range(1, len(joint_detected)):
                total_transitions += 1
                if joint_detected[i] != joint_detected[i-1]:
                    flip_count += 1

        flip_rate = 100 * flip_count / total_transitions if total_transitions > 0 else 0
        row += f" {flip_rate:>13.2f}% |"

    print(row)

# Hypothesis 3: Is the jitter problem worse at certain rotations?
print("\n\n### Hypothesis 3: Jitter by Rotation (checking if lateral view causes instability)")

print("\n| Rotation | MP Jitter | MP Joint Changes | MN Jitter | MN Joint Changes |")
print("|----------|-----------|------------------|-----------|------------------|")

for bucket in ['0-10', '10-20', '30-40', '60-70', '80-90']:
    row = f"| {bucket:>8}° |"

    for model in ['MediaPipe', 'MoveNet']:
        model_data = df_clean[(df_clean['model'] == model) & (df_clean['rotation_bucket'] == bucket)].copy()

        jitters = []
        joint_changes = 0
        total = 0

        for video_id in model_data['video_id'].unique():
            video_data = model_data[model_data['video_id'] == video_id].sort_values('frame_idx')

            if len(video_data) < 2:
                continue

            nmpjpe = video_data['nmpjpe'].values
            valid_joints = video_data['valid_joints'].values

            for i in range(1, len(nmpjpe)):
                jitters.append(abs(nmpjpe[i] - nmpjpe[i-1]))
                total += 1
                if valid_joints[i] != valid_joints[i-1]:
                    joint_changes += 1

        mean_jitter = np.mean(jitters) if jitters else 0
        change_rate = 100 * joint_changes / total if total > 0 else 0

        row += f" {mean_jitter:>9.2f}% | {change_rate:>16.1f}% |"

    print(row)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
Key findings:
1. When valid_joints changes between frames, jitter is MUCH higher
2. MediaPipe has more joint detection instability (joints appearing/disappearing)
3. Right-side joints (especially elbow, wrist) flip most often for MediaPipe
4. This explains MediaPipe's higher jitter: it's not the pose estimation itself,
   but the instability of which joints are detected frame-to-frame.
""")
