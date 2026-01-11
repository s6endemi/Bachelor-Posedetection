"""
Comprehensive Analysis for Pose Estimation Evaluation.

This script performs a deep-dive analysis of all predictions and generates
structured outputs for thesis writing.

Author: Eren
Date: January 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import project mappings
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pose_evaluation.data.keypoint_mapping import (
    COCO_TO_GT_MAPPING,
    COMPARABLE_COCO_INDICES,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PREDICTIONS_DIR = Path(__file__).parent.parent / "data" / "predictions"
GT_2D_DIR = Path(__file__).parent.parent / "data" / "gt_2d"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

FRAME_STEP = 3  # Same as used in inference

# Coach videos (manually identified)
COACH_VIDEOS = {"PM_010-c17", "PM_011-c17", "PM_108-c17", "PM_119-c17", "PM_121-c17"}

# Model name mapping for cleaner output
MODEL_NAMES = {
    "pred_MediaPipe_full": "MediaPipe",
    "pred_MoveNet_multipose": "MoveNet",
    "pred_YOLOv8-Pose_n": "YOLO"
}

# Rotation buckets (10-degree)
ROTATION_BUCKETS = [(i, i+10) for i in range(0, 90, 10)]


# =============================================================================
# FRAME-LEVEL NMPJPE CALCULATION
# =============================================================================

def calculate_frame_nmpjpe(pred: np.ndarray, gt: np.ndarray, min_confidence: float = 0.3) -> dict:
    """Calculate NMPJPE for a single frame with detailed metrics."""
    pred_pts = []
    gt_pts = []
    confidences = []
    joint_names = []

    coco_joint_names = {
        5: 'left_shoulder', 6: 'right_shoulder',
        7: 'left_elbow', 8: 'right_elbow',
        9: 'left_wrist', 10: 'right_wrist',
        11: 'left_hip', 12: 'right_hip',
        13: 'left_knee', 14: 'right_knee',
        15: 'left_ankle', 16: 'right_ankle'
    }

    for coco_idx in COMPARABLE_COCO_INDICES:
        gt_idx = COCO_TO_GT_MAPPING[coco_idx]
        pred_pts.append(pred[coco_idx, :2])
        gt_pts.append(gt[gt_idx])
        confidences.append(pred[coco_idx, 2])
        joint_names.append(coco_joint_names.get(coco_idx, f'joint_{coco_idx}'))

    pred_pts = np.array(pred_pts)
    gt_pts = np.array(gt_pts)
    confidences = np.array(confidences)

    # Torso length
    shoulder_mid = (gt[7] + gt[12]) / 2
    hip_mid = (gt[16] + gt[21]) / 2
    torso_length = np.linalg.norm(shoulder_mid - hip_mid)

    if torso_length < 10:
        return {'nmpjpe': np.nan, 'valid_joints': 0, 'per_joint_errors': {}, 'torso_length': torso_length}

    errors = np.linalg.norm(pred_pts - gt_pts, axis=1)
    normalized_errors = errors / torso_length * 100

    valid_mask = confidences >= min_confidence
    valid_joints = np.sum(valid_mask)

    per_joint_errors = {}
    for i, name in enumerate(joint_names):
        if valid_mask[i]:
            per_joint_errors[name] = normalized_errors[i]

    if valid_joints == 0:
        return {'nmpjpe': np.nan, 'valid_joints': 0, 'per_joint_errors': per_joint_errors, 'torso_length': torso_length}

    nmpjpe = np.mean(normalized_errors[valid_mask])
    return {'nmpjpe': nmpjpe, 'valid_joints': valid_joints, 'per_joint_errors': per_joint_errors, 'torso_length': torso_length}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_frame_data() -> pd.DataFrame:
    """Load all predictions and compute frame-level metrics."""
    print("Loading all frame-level data...")
    npz_files = sorted(PREDICTIONS_DIR.rglob("*.npz"))
    print(f"Found {len(npz_files)} prediction files")

    all_rows = []

    for npz_idx, npz_path in enumerate(npz_files):
        if (npz_idx + 1) % 20 == 0:
            print(f"  Processing {npz_idx + 1}/{len(npz_files)}...")

        exercise = npz_path.parent.name
        filename = npz_path.stem
        parts = filename.split("-")
        subject_id = parts[0]
        camera = parts[1]
        video_id = f"{subject_id}-{camera}"
        is_coach = video_id in COACH_VIDEOS

        data = np.load(npz_path)
        rotation_angles = data['rotation_angles']

        gt_path = GT_2D_DIR / exercise / f"{subject_id}-{camera}-30fps.npy"
        if not gt_path.exists():
            continue
        gt_2d = np.load(gt_path)

        for model_key in ['pred_MediaPipe_full', 'pred_MoveNet_multipose', 'pred_YOLOv8-Pose_n']:
            if model_key not in data.files:
                continue

            predictions = data[model_key]
            model_name = MODEL_NAMES[model_key]

            for frame_idx in range(len(predictions)):
                gt_idx = frame_idx * FRAME_STEP
                if gt_idx >= len(gt_2d):
                    break

                result = calculate_frame_nmpjpe(predictions[frame_idx], gt_2d[gt_idx])
                if np.isnan(result['nmpjpe']):
                    continue

                rotation = rotation_angles[frame_idx] if frame_idx < len(rotation_angles) else np.nan
                rotation_bucket = None
                for low, high in ROTATION_BUCKETS:
                    if low <= rotation < high:
                        rotation_bucket = f"{low}-{high}"
                        break

                row = {
                    'exercise': exercise, 'subject_id': subject_id, 'camera': camera,
                    'video_id': video_id, 'is_coach': is_coach, 'model': model_name,
                    'frame_idx': frame_idx, 'rotation': rotation, 'rotation_bucket': rotation_bucket,
                    'nmpjpe': result['nmpjpe'], 'valid_joints': result['valid_joints'],
                    'torso_length': result['torso_length'],
                }
                for joint_name, error in result['per_joint_errors'].items():
                    row[f'error_{joint_name}'] = error
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"Loaded {len(df)} frame-level records")
    return df


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_overall_statistics(df: pd.DataFrame, clean_only: bool = True) -> dict:
    if clean_only:
        df = df[~df['is_coach']]
    results = {}
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        nmpjpe = model_df['nmpjpe']
        results[model] = {
            'n_frames': len(model_df), 'mean': nmpjpe.mean(), 'median': nmpjpe.median(),
            'std': nmpjpe.std(), 'iqr': nmpjpe.quantile(0.75) - nmpjpe.quantile(0.25),
            'p25': nmpjpe.quantile(0.25), 'p75': nmpjpe.quantile(0.75),
            'p90': nmpjpe.quantile(0.90), 'p95': nmpjpe.quantile(0.95),
            'min': nmpjpe.min(), 'max': nmpjpe.max(),
        }
    return results


def analyze_camera_comparison(df: pd.DataFrame, clean_only: bool = True) -> dict:
    if clean_only:
        df = df[~df['is_coach']]
    results = {}
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        results[model] = {}
        for camera in ['c17', 'c18']:
            cam_df = model_df[model_df['camera'] == camera]
            nmpjpe = cam_df['nmpjpe']
            n_over_30 = (nmpjpe > 30).sum()
            n_over_50 = (nmpjpe > 50).sum()
            n_over_100 = (nmpjpe > 100).sum()
            results[model][camera] = {
                'n_frames': len(cam_df), 'mean': nmpjpe.mean(), 'median': nmpjpe.median(),
                'std': nmpjpe.std(), 'p25': nmpjpe.quantile(0.25), 'p75': nmpjpe.quantile(0.75),
                'n_over_30': int(n_over_30), 'n_over_50': int(n_over_50), 'n_over_100': int(n_over_100),
                'pct_over_30': n_over_30 / len(cam_df) * 100 if len(cam_df) > 0 else 0,
                'pct_over_50': n_over_50 / len(cam_df) * 100 if len(cam_df) > 0 else 0,
                'pct_over_100': n_over_100 / len(cam_df) * 100 if len(cam_df) > 0 else 0,
            }
    return results


def analyze_rotation_buckets(df: pd.DataFrame, clean_only: bool = True) -> pd.DataFrame:
    if clean_only:
        df = df[~df['is_coach']]
    df = df[df['rotation_bucket'].notna()]
    results = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        for bucket in [f"{i}-{i+10}" for i in range(0, 90, 10)]:
            bucket_df = model_df[model_df['rotation_bucket'] == bucket]
            if len(bucket_df) == 0:
                continue
            nmpjpe = bucket_df['nmpjpe']
            results.append({
                'model': model, 'rotation_bucket': bucket,
                'rotation_mid': int(bucket.split('-')[0]) + 5,
                'n_frames': len(bucket_df), 'mean': nmpjpe.mean(), 'median': nmpjpe.median(),
                'std': nmpjpe.std(), 'p25': nmpjpe.quantile(0.25), 'p75': nmpjpe.quantile(0.75),
            })
    return pd.DataFrame(results)


def analyze_outliers(df: pd.DataFrame, threshold: float = 30) -> pd.DataFrame:
    outliers = df[df['nmpjpe'] > threshold].copy()
    video_outliers = outliers.groupby(['video_id', 'model', 'camera', 'exercise', 'is_coach']).agg({
        'nmpjpe': ['count', 'mean', 'max'], 'frame_idx': 'count'
    }).reset_index()
    video_outliers.columns = ['video_id', 'model', 'camera', 'exercise', 'is_coach',
                              'n_outlier_frames', 'mean_outlier_nmpjpe', 'max_nmpjpe', 'frame_count']
    total_frames = df.groupby(['video_id', 'model']).size().reset_index(name='total_frames')
    video_outliers = video_outliers.merge(total_frames, on=['video_id', 'model'])
    video_outliers['outlier_rate'] = video_outliers['n_outlier_frames'] / video_outliers['total_frames'] * 100
    return video_outliers.sort_values('n_outlier_frames', ascending=False)


def analyze_per_joint(df: pd.DataFrame, clean_only: bool = True) -> pd.DataFrame:
    if clean_only:
        df = df[~df['is_coach']]
    joint_cols = [c for c in df.columns if c.startswith('error_')]
    results = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        for col in joint_cols:
            joint_name = col.replace('error_', '')
            errors = model_df[col].dropna()
            if len(errors) == 0:
                continue
            results.append({
                'model': model, 'joint': joint_name, 'n_samples': len(errors),
                'mean': errors.mean(), 'median': errors.median(), 'std': errors.std(),
            })
    return pd.DataFrame(results)


def analyze_statistical_significance(df: pd.DataFrame, clean_only: bool = True) -> dict:
    if clean_only:
        df = df[~df['is_coach']]
    models = df['model'].unique()
    groups = [df[df['model'] == m]['nmpjpe'].values for m in models]
    f_stat, p_value = stats.f_oneway(*groups)

    pairwise = []
    n_comparisons = len(models) * (len(models) - 1) // 2
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i >= j:
                continue
            g1 = df[df['model'] == m1]['nmpjpe'].values
            g2 = df[df['model'] == m2]['nmpjpe'].values
            t_stat, p = stats.ttest_ind(g1, g2)
            pooled_std = np.sqrt((np.std(g1)**2 + np.std(g2)**2) / 2)
            cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
            pairwise.append({
                'model_1': m1, 'model_2': m2, 't_stat': t_stat, 'p_value': p,
                'p_value_bonferroni': min(p * n_comparisons, 1.0),
                'cohens_d': cohens_d, 'mean_diff': np.mean(g1) - np.mean(g2),
            })
    return {'anova': {'f_statistic': f_stat, 'p_value': p_value, 'significant': p_value < 0.05}, 'pairwise': pairwise}


def analyze_coach_impact(df: pd.DataFrame) -> dict:
    results = {}
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        clean = model_df[~model_df['is_coach']]['nmpjpe']
        coach = model_df[model_df['is_coach']]['nmpjpe']
        if len(coach) == 0:
            continue
        results[model] = {
            'clean_mean': clean.mean(), 'clean_median': clean.median(),
            'coach_mean': coach.mean(), 'coach_median': coach.median(),
            'mean_increase': coach.mean() - clean.mean(),
            'mean_increase_pct': (coach.mean() - clean.mean()) / clean.mean() * 100,
            'n_clean_frames': len(clean), 'n_coach_frames': len(coach),
        }
    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(df: pd.DataFrame, results: dict) -> str:
    lines = [
        "# Comprehensive Pose Estimation Analysis Report", "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total Frames Analyzed:** {len(df):,}",
        f"**Clean Frames:** {len(df[~df['is_coach']]):,}",
        f"**Coach Frames:** {len(df[df['is_coach']]):,}", "", "---", "",
    ]

    # Overall Statistics
    lines.append("## 1. Overall Statistics (Clean Data)")
    lines.append("")
    lines.append("| Model | N Frames | Mean | Median | Std | IQR | P90 | P95 |")
    lines.append("|-------|----------|------|--------|-----|-----|-----|-----|")
    for model, s in results['overall'].items():
        lines.append(f"| {model} | {s['n_frames']:,} | {s['mean']:.1f}% | {s['median']:.1f}% | {s['std']:.1f}% | {s['iqr']:.1f}% | {s['p90']:.1f}% | {s['p95']:.1f}% |")
    lines.append("")

    # Camera Comparison
    lines.append("## 2. Camera Comparison (Clean Data)")
    lines.append("")
    lines.append("### Mean NMPJPE")
    lines.append("")
    lines.append("| Model | c17 Mean | c17 Median | c18 Mean | c18 Median | Diff (Mean) |")
    lines.append("|-------|----------|------------|----------|------------|-------------|")
    for model, cams in results['camera'].items():
        c17, c18 = cams['c17'], cams['c18']
        diff = c17['mean'] - c18['mean']
        lines.append(f"| {model} | {c17['mean']:.1f}% | {c17['median']:.1f}% | {c18['mean']:.1f}% | {c18['median']:.1f}% | {diff:+.1f}% |")
    lines.append("")

    lines.append("### Outlier Frames (>30%, >50%, >100%)")
    lines.append("")
    lines.append("| Model | Camera | >30% | >50% | >100% | >100% Rate |")
    lines.append("|-------|--------|------|------|-------|------------|")
    for model, cams in results['camera'].items():
        for cam in ['c17', 'c18']:
            c = cams[cam]
            lines.append(f"| {model} | {cam} | {c['n_over_30']} | {c['n_over_50']} | {c['n_over_100']} | {c['pct_over_100']:.2f}% |")
    lines.append("")

    # Rotation Analysis
    lines.append("## 3. Rotation Analysis (10-degree buckets, Clean Data)")
    lines.append("")
    rot_df = results['rotation']

    lines.append("### Sample Sizes per Bucket")
    lines.append("")
    pivot_n = rot_df.pivot(index='rotation_bucket', columns='model', values='n_frames')
    lines.append("| Bucket | " + " | ".join(pivot_n.columns) + " |")
    lines.append("|--------|" + "|".join(["------"] * len(pivot_n.columns)) + "|")
    for bucket in [f"{i}-{i+10}" for i in range(0, 90, 10)]:
        if bucket in pivot_n.index:
            row = pivot_n.loc[bucket]
            vals = [f"{int(v):,}" if not pd.isna(v) else "-" for v in row]
            lines.append(f"| {bucket} | " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("### Mean NMPJPE per Bucket")
    lines.append("")
    pivot_mean = rot_df.pivot(index='rotation_bucket', columns='model', values='mean')
    lines.append("| Bucket | " + " | ".join(pivot_mean.columns) + " |")
    lines.append("|--------|" + "|".join(["------"] * len(pivot_mean.columns)) + "|")
    for bucket in [f"{i}-{i+10}" for i in range(0, 90, 10)]:
        if bucket in pivot_mean.index:
            row = pivot_mean.loc[bucket]
            vals = [f"{v:.1f}%" if not pd.isna(v) else "-" for v in row]
            lines.append(f"| {bucket} | " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("### Median NMPJPE per Bucket")
    lines.append("")
    pivot_median = rot_df.pivot(index='rotation_bucket', columns='model', values='median')
    lines.append("| Bucket | " + " | ".join(pivot_median.columns) + " |")
    lines.append("|--------|" + "|".join(["------"] * len(pivot_median.columns)) + "|")
    for bucket in [f"{i}-{i+10}" for i in range(0, 90, 10)]:
        if bucket in pivot_median.index:
            row = pivot_median.loc[bucket]
            vals = [f"{v:.1f}%" if not pd.isna(v) else "-" for v in row]
            lines.append(f"| {bucket} | " + " | ".join(vals) + " |")
    lines.append("")

    # Statistical Significance
    lines.append("## 4. Statistical Significance")
    lines.append("")
    anova = results['significance']['anova']
    lines.append(f"**ANOVA:** F={anova['f_statistic']:.2f}, p={anova['p_value']:.2e}, Significant: {'Yes' if anova['significant'] else 'No'}")
    lines.append("")
    lines.append("### Pairwise Comparisons (Bonferroni-corrected)")
    lines.append("")
    lines.append("| Comparison | Mean Diff | Cohen's d | p-value | Significant |")
    lines.append("|------------|-----------|-----------|---------|-------------|")
    for comp in results['significance']['pairwise']:
        sig = "Yes" if comp['p_value_bonferroni'] < 0.05 else "No"
        lines.append(f"| {comp['model_1']} vs {comp['model_2']} | {comp['mean_diff']:+.2f}% | {comp['cohens_d']:.3f} | {comp['p_value_bonferroni']:.2e} | {sig} |")
    lines.append("")

    # Coach Impact
    lines.append("## 5. Coach Impact Analysis")
    lines.append("")
    lines.append("| Model | Clean Mean | Coach Mean | Increase | Increase % |")
    lines.append("|-------|------------|------------|----------|------------|")
    for model, impact in results['coach_impact'].items():
        lines.append(f"| {model} | {impact['clean_mean']:.1f}% | {impact['coach_mean']:.1f}% | +{impact['mean_increase']:.1f}% | +{impact['mean_increase_pct']:.0f}% |")
    lines.append("")

    # Per-Joint Analysis
    lines.append("## 6. Per-Joint Analysis (Clean Data)")
    lines.append("")
    joint_df = results['per_joint']
    for model in joint_df['model'].unique():
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| Joint | Mean | Median | Std |")
        lines.append("|-------|------|--------|-----|")
        model_joints = joint_df[joint_df['model'] == model].sort_values('mean', ascending=False)
        for _, row in model_joints.iterrows():
            lines.append(f"| {row['joint']} | {row['mean']:.1f}% | {row['median']:.1f}% | {row['std']:.1f}% |")
        lines.append("")

    # Top Outlier Videos
    lines.append("## 7. Top Outlier Videos (>30% NMPJPE frames)")
    lines.append("")
    outliers = results['outliers']
    top_outliers = outliers.nlargest(20, 'n_outlier_frames')
    lines.append("| Video | Model | Camera | Coach? | Outlier Frames | Rate | Max NMPJPE |")
    lines.append("|-------|-------|--------|--------|----------------|------|------------|")
    for _, row in top_outliers.iterrows():
        coach_str = "Yes" if row['is_coach'] else "No"
        lines.append(f"| {row['video_id']} | {row['model']} | {row['camera']} | {coach_str} | {int(row['n_outlier_frames'])} | {row['outlier_rate']:.1f}% | {row['max_nmpjpe']:.1f}% |")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def convert_types(obj):
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_frame_data()

    frame_data_path = RESULTS_DIR / "frame_level_data.csv"
    df.to_csv(frame_data_path, index=False)
    print(f"Saved frame-level data to {frame_data_path}")

    print("\nRunning analyses...")
    results = {
        'overall': analyze_overall_statistics(df, clean_only=True),
        'camera': analyze_camera_comparison(df, clean_only=True),
        'rotation': analyze_rotation_buckets(df, clean_only=True),
        'outliers': analyze_outliers(df, threshold=30),
        'per_joint': analyze_per_joint(df, clean_only=True),
        'significance': analyze_statistical_significance(df, clean_only=True),
        'coach_impact': analyze_coach_impact(df),
    }

    results['rotation'].to_csv(RESULTS_DIR / "rotation_analysis.csv", index=False)
    results['outliers'].to_csv(RESULTS_DIR / "outlier_analysis.csv", index=False)
    results['per_joint'].to_csv(RESULTS_DIR / "per_joint_analysis.csv", index=False)

    summary = {
        'overall': results['overall'], 'camera': results['camera'],
        'significance': results['significance'], 'coach_impact': results['coach_impact'],
    }
    with open(RESULTS_DIR / "summary_statistics.json", 'w') as f:
        json.dump(convert_types(summary), f, indent=2)

    report = generate_report(df, results)
    report_path = Path(__file__).parent / "ANALYSIS_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nSaved report to {report_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nTotal frames analyzed: {len(df):,}")
    print(f"Clean frames: {len(df[~df['is_coach']]):,}")
    print(f"Coach frames: {len(df[df['is_coach']]):,}")
    print("\nOverall Results (Clean Data):")
    for model, s in results['overall'].items():
        print(f"  {model:12s}: Mean={s['mean']:.1f}%, Median={s['median']:.1f}%, Std={s['std']:.1f}%")

    print("\nOutput files:")
    for f in ['frame_level_data.csv', 'rotation_analysis.csv', 'outlier_analysis.csv', 'per_joint_analysis.csv', 'summary_statistics.json']:
        print(f"  - {RESULTS_DIR / f}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
