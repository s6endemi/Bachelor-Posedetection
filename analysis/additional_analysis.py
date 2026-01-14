"""
Additional Analysis: Temporal Jitter and Valid-Joints Correlation
Adds two new analyses to the evaluation results.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Paths
RESULTS_DIR = Path(__file__).parent / "results"
FRAME_DATA = RESULTS_DIR / "frame_level_data.csv"

def load_data():
    """Load frame-level data."""
    print("Loading frame-level data...")
    df = pd.read_csv(FRAME_DATA)
    print(f"Loaded {len(df):,} frames")
    return df

def analyze_temporal_jitter(df):
    """
    Analyze frame-to-frame stability (jitter).
    Lower jitter = more stable predictions = better for real-time feedback.
    """
    print("\n" + "="*70)
    print("TEMPORAL JITTER ANALYSIS")
    print("="*70)

    # Filter to clean data (no coach videos, no extreme outliers)
    df_clean = df[(df['is_coach'] == False) & (df['nmpjpe'] < 100)].copy()

    results = []

    for model in ['MediaPipe', 'MoveNet', 'YOLO']:
        model_data = df_clean[df_clean['model'] == model].copy()

        jitter_values = []

        # Calculate jitter per video
        for video_id in model_data['video_id'].unique():
            video_data = model_data[model_data['video_id'] == video_id].sort_values('frame_idx')

            if len(video_data) < 2:
                continue

            # Frame-to-frame absolute difference in NMPJPE
            nmpjpe_values = video_data['nmpjpe'].values
            frame_diffs = np.abs(np.diff(nmpjpe_values))

            jitter_values.extend(frame_diffs)

        jitter_array = np.array(jitter_values)

        results.append({
            'model': model,
            'mean_jitter': np.mean(jitter_array),
            'median_jitter': np.median(jitter_array),
            'std_jitter': np.std(jitter_array),
            'p90_jitter': np.percentile(jitter_array, 90),
            'p95_jitter': np.percentile(jitter_array, 95),
            'n_transitions': len(jitter_array)
        })

        print(f"\n{model}:")
        print(f"  Mean Jitter:   {np.mean(jitter_array):.2f}%")
        print(f"  Median Jitter: {np.median(jitter_array):.2f}%")
        print(f"  Std Jitter:    {np.std(jitter_array):.2f}%")
        print(f"  P90 Jitter:    {np.percentile(jitter_array, 90):.2f}%")
        print(f"  P95 Jitter:    {np.percentile(jitter_array, 95):.2f}%")
        print(f"  N Transitions: {len(jitter_array):,}")

    # Save results
    jitter_df = pd.DataFrame(results)
    jitter_df.to_csv(RESULTS_DIR / "temporal_jitter.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'temporal_jitter.csv'}")

    # Statistical comparison
    print("\n### Statistical Comparison (Jitter)")

    model_jitters = {}
    for model in ['MediaPipe', 'MoveNet', 'YOLO']:
        model_data = df_clean[df_clean['model'] == model].copy()
        jitter_values = []
        for video_id in model_data['video_id'].unique():
            video_data = model_data[model_data['video_id'] == video_id].sort_values('frame_idx')
            if len(video_data) >= 2:
                nmpjpe_values = video_data['nmpjpe'].values
                jitter_values.extend(np.abs(np.diff(nmpjpe_values)))
        model_jitters[model] = np.array(jitter_values)

    # Pairwise comparisons
    comparisons = [
        ('MediaPipe', 'MoveNet'),
        ('MediaPipe', 'YOLO'),
        ('MoveNet', 'YOLO')
    ]

    print("\n| Comparison | Mean Diff | t-stat | p-value | Significant |")
    print("|------------|-----------|--------|---------|-------------|")

    for m1, m2 in comparisons:
        t_stat, p_val = stats.ttest_ind(model_jitters[m1], model_jitters[m2])
        mean_diff = np.mean(model_jitters[m1]) - np.mean(model_jitters[m2])
        sig = "Yes" if p_val < 0.05 else "No"
        print(f"| {m1} vs {m2} | {mean_diff:+.3f}% | {t_stat:.2f} | {p_val:.4f} | {sig} |")

    return jitter_df

def analyze_jitter_by_conditions(df):
    """Analyze jitter by camera, rotation, and exercise."""
    print("\n" + "="*70)
    print("JITTER BY CONDITIONS")
    print("="*70)

    df_clean = df[(df['is_coach'] == False) & (df['nmpjpe'] < 100)].copy()

    def calc_jitter_for_subset(subset):
        """Calculate mean jitter for a data subset."""
        jitter_values = []
        for video_id in subset['video_id'].unique():
            video_data = subset[subset['video_id'] == video_id].sort_values('frame_idx')
            if len(video_data) >= 2:
                nmpjpe_values = video_data['nmpjpe'].values
                jitter_values.extend(np.abs(np.diff(nmpjpe_values)))
        return np.mean(jitter_values) if jitter_values else np.nan

    # By Camera
    print("\n### Jitter by Camera")
    print("\n| Camera | MediaPipe | MoveNet | YOLO |")
    print("|--------|-----------|---------|------|")
    for cam in ['c17', 'c18']:
        row = f"| {cam} |"
        for model in ['MediaPipe', 'MoveNet', 'YOLO']:
            subset = df_clean[(df_clean['camera'] == cam) & (df_clean['model'] == model)]
            jitter = calc_jitter_for_subset(subset)
            row += f" {jitter:.2f}% |"
        print(row)

    # By Rotation Bucket
    print("\n### Jitter by Rotation")
    print("\n| Rotation | MediaPipe | MoveNet | YOLO |")
    print("|----------|-----------|---------|------|")
    for bucket in ['0-10', '10-20', '20-30', '30-40', '50-60', '60-70', '70-80', '80-90']:
        row = f"| {bucket}° |"
        for model in ['MediaPipe', 'MoveNet', 'YOLO']:
            subset = df_clean[(df_clean['rotation_bucket'] == bucket) & (df_clean['model'] == model)]
            jitter = calc_jitter_for_subset(subset)
            if np.isnan(jitter):
                row += " - |"
            else:
                row += f" {jitter:.2f}% |"
        print(row)

def analyze_valid_joints_correlation(df):
    """
    Analyze correlation between number of valid joints and error.
    This is a proxy for confidence - fewer joints detected = lower confidence.
    """
    print("\n" + "="*70)
    print("VALID JOINTS VS ERROR ANALYSIS")
    print("="*70)

    # Filter to clean data
    df_clean = df[(df['is_coach'] == False) & (df['nmpjpe'] < 100)].copy()

    results = []

    for model in ['MediaPipe', 'MoveNet', 'YOLO']:
        model_data = df_clean[df_clean['model'] == model]

        # Correlation
        corr, p_val = stats.pearsonr(model_data['valid_joints'], model_data['nmpjpe'])

        print(f"\n{model}:")
        print(f"  Pearson r: {corr:.4f} (p={p_val:.2e})")

        # Group by valid_joints
        print(f"\n  Error by Valid Joints:")
        print(f"  | Valid Joints | N Frames | Mean NMPJPE | Median NMPJPE |")
        print(f"  |--------------|----------|-------------|---------------|")

        for n_joints in sorted(model_data['valid_joints'].unique(), reverse=True):
            subset = model_data[model_data['valid_joints'] == n_joints]
            print(f"  | {n_joints:>12} | {len(subset):>8,} | {subset['nmpjpe'].mean():>11.2f}% | {subset['nmpjpe'].median():>13.2f}% |")

            results.append({
                'model': model,
                'valid_joints': n_joints,
                'n_frames': len(subset),
                'mean_nmpjpe': subset['nmpjpe'].mean(),
                'median_nmpjpe': subset['nmpjpe'].median(),
                'std_nmpjpe': subset['nmpjpe'].std()
            })

    # Save results
    valid_joints_df = pd.DataFrame(results)
    valid_joints_df.to_csv(RESULTS_DIR / "valid_joints_analysis.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'valid_joints_analysis.csv'}")

    # Summary table
    print("\n### Summary: Can we use Valid Joints as Quality Filter?")
    print("\n| Model | Frames with <12 Joints | Mean Error (<12) | Mean Error (12) | Diff |")
    print("|-------|------------------------|------------------|-----------------|------|")

    for model in ['MediaPipe', 'MoveNet', 'YOLO']:
        model_data = df_clean[df_clean['model'] == model]

        full_joints = model_data[model_data['valid_joints'] == 12]
        partial_joints = model_data[model_data['valid_joints'] < 12]

        n_partial = len(partial_joints)
        pct_partial = 100 * n_partial / len(model_data)
        mean_partial = partial_joints['nmpjpe'].mean() if len(partial_joints) > 0 else 0
        mean_full = full_joints['nmpjpe'].mean()
        diff = mean_partial - mean_full

        print(f"| {model} | {n_partial:,} ({pct_partial:.1f}%) | {mean_partial:.1f}% | {mean_full:.1f}% | +{diff:.1f}% |")

    return valid_joints_df

def analyze_joint_detection_rate(df):
    """Analyze which joints are most often missing."""
    print("\n" + "="*70)
    print("JOINT DETECTION RATE ANALYSIS")
    print("="*70)

    df_clean = df[(df['is_coach'] == False)].copy()

    joint_cols = [
        'error_left_shoulder', 'error_right_shoulder',
        'error_left_elbow', 'error_right_elbow',
        'error_left_wrist', 'error_right_wrist',
        'error_left_hip', 'error_right_hip',
        'error_left_knee', 'error_right_knee',
        'error_left_ankle', 'error_right_ankle'
    ]

    print("\n### Joint Detection Rate (% of frames where joint is detected)")
    print("\n| Joint | MediaPipe | MoveNet | YOLO |")
    print("|-------|-----------|---------|------|")

    results = []

    for joint in joint_cols:
        joint_name = joint.replace('error_', '').replace('_', ' ').title()
        row = f"| {joint_name} |"

        for model in ['MediaPipe', 'MoveNet', 'YOLO']:
            model_data = df_clean[df_clean['model'] == model]
            detected = model_data[joint].notna().sum()
            total = len(model_data)
            rate = 100 * detected / total
            row += f" {rate:.1f}% |"

            results.append({
                'joint': joint_name,
                'model': model,
                'detection_rate': rate,
                'n_detected': detected,
                'n_total': total
            })

        print(row)

    # Save
    detection_df = pd.DataFrame(results)
    detection_df.to_csv(RESULTS_DIR / "joint_detection_rate.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'joint_detection_rate.csv'}")

    return detection_df

def main():
    """Run all additional analyses."""
    df = load_data()

    # 1. Temporal Jitter
    jitter_df = analyze_temporal_jitter(df)

    # 2. Jitter by conditions
    analyze_jitter_by_conditions(df)

    # 3. Valid Joints vs Error
    valid_joints_df = analyze_valid_joints_correlation(df)

    # 4. Joint Detection Rate
    detection_df = analyze_joint_detection_rate(df)

    print("\n" + "="*70)
    print("ADDITIONAL ANALYSIS COMPLETE")
    print("="*70)
    print("\nNew files created:")
    print("  - temporal_jitter.csv")
    print("  - valid_joints_analysis.csv")
    print("  - joint_detection_rate.csv")

if __name__ == "__main__":
    main()
