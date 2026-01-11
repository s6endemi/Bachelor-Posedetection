"""
Extended Analysis - Zusätzliche Analysen basierend auf Frame-Level Daten.

Ergänzt comprehensive_analysis.py um:
1. Übungs-Analyse (Ex1-Ex6)
2. Rotation getrennt nach Kamera (c17-only, c18-only)
3. Analyse OHNE >100% Frames (Person-Switch-bereinigt)
4. Vergleich: Raw vs Bereinigt

Author: Eren
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_frame_data() -> pd.DataFrame:
    """Load pre-computed frame-level data."""
    return pd.read_csv(RESULTS_DIR / "frame_level_data.csv")


# =============================================================================
# 1. ÜBUNGS-ANALYSE (Ex1-Ex6)
# =============================================================================

def analyze_by_exercise(df: pd.DataFrame, clean_only: bool = True, filter_extreme: bool = False):
    """Analyze NMPJPE by exercise."""
    if clean_only:
        df = df[~df['is_coach']]
    if filter_extreme:
        df = df[df['nmpjpe'] <= 100]

    print("\n" + "=" * 70)
    print("ÜBUNGS-ANALYSE" + (" (ohne >100% Frames)" if filter_extreme else ""))
    print("=" * 70)

    exercises = sorted(df['exercise'].unique())

    # Mean
    print("\n### Mean NMPJPE pro Übung")
    print("\n| Exercise | MediaPipe | MoveNet | YOLO | N Frames |")
    print("|----------|-----------|---------|------|----------|")

    for ex in exercises:
        ex_df = df[df['exercise'] == ex]
        n_frames = len(ex_df) // 3  # Approx per model

        mp = ex_df[ex_df['model'] == 'MediaPipe']['nmpjpe'].mean()
        mn = ex_df[ex_df['model'] == 'MoveNet']['nmpjpe'].mean()
        yolo = ex_df[ex_df['model'] == 'YOLO']['nmpjpe'].mean()

        print(f"| {ex} | {mp:.1f}% | {mn:.1f}% | {yolo:.1f}% | ~{n_frames:,} |")

    # Median
    print("\n### Median NMPJPE pro Übung")
    print("\n| Exercise | MediaPipe | MoveNet | YOLO |")
    print("|----------|-----------|---------|------|")

    for ex in exercises:
        ex_df = df[df['exercise'] == ex]

        mp = ex_df[ex_df['model'] == 'MediaPipe']['nmpjpe'].median()
        mn = ex_df[ex_df['model'] == 'MoveNet']['nmpjpe'].median()
        yolo = ex_df[ex_df['model'] == 'YOLO']['nmpjpe'].median()

        print(f"| {ex} | {mp:.1f}% | {mn:.1f}% | {yolo:.1f}% |")


# =============================================================================
# 2. ROTATION GETRENNT NACH KAMERA
# =============================================================================

def analyze_rotation_by_camera(df: pd.DataFrame, clean_only: bool = True, filter_extreme: bool = False):
    """Analyze rotation effect separately for each camera."""
    if clean_only:
        df = df[~df['is_coach']]
    if filter_extreme:
        df = df[df['nmpjpe'] <= 100]

    df = df[df['rotation_bucket'].notna()]

    print("\n" + "=" * 70)
    print("ROTATION NACH KAMERA" + (" (ohne >100% Frames)" if filter_extreme else ""))
    print("=" * 70)

    buckets = [f"{i}-{i+10}" for i in range(0, 90, 10)]

    for camera in ['c17', 'c18']:
        cam_df = df[df['camera'] == camera]

        print(f"\n### {camera.upper()} - Median NMPJPE")
        print("\n| Bucket | MediaPipe | MoveNet | YOLO | N Frames |")
        print("|--------|-----------|---------|------|----------|")

        for bucket in buckets:
            bucket_df = cam_df[cam_df['rotation_bucket'] == bucket]
            if len(bucket_df) == 0:
                continue

            n_frames = len(bucket_df) // 3

            mp = bucket_df[bucket_df['model'] == 'MediaPipe']['nmpjpe'].median()
            mn = bucket_df[bucket_df['model'] == 'MoveNet']['nmpjpe'].median()
            yolo = bucket_df[bucket_df['model'] == 'YOLO']['nmpjpe'].median()

            # Handle NaN
            mp_str = f"{mp:.1f}%" if not pd.isna(mp) else "-"
            mn_str = f"{mn:.1f}%" if not pd.isna(mn) else "-"
            yolo_str = f"{yolo:.1f}%" if not pd.isna(yolo) else "-"

            print(f"| {bucket} | {mp_str} | {mn_str} | {yolo_str} | ~{n_frames:,} |")


# =============================================================================
# 3. ANALYSE MIT/OHNE EXTREME FRAMES
# =============================================================================

def compare_with_without_extreme(df: pd.DataFrame, clean_only: bool = True):
    """Compare statistics with and without >100% frames."""
    if clean_only:
        df = df[~df['is_coach']]

    print("\n" + "=" * 70)
    print("VERGLEICH: MIT vs OHNE >100% FRAMES (Person-Switch-Bereinigung)")
    print("=" * 70)

    df_raw = df
    df_clean = df[df['nmpjpe'] <= 100]

    n_removed = len(df_raw) - len(df_clean)
    pct_removed = n_removed / len(df_raw) * 100

    print(f"\nEntfernte Frames: {n_removed:,} ({pct_removed:.2f}%)")

    print("\n### Overall Statistics")
    print("\n| Model | Raw Mean | Clean Mean | Diff | Raw Median | Clean Median | Diff |")
    print("|-------|----------|------------|------|------------|--------------|------|")

    for model in df['model'].unique():
        raw = df_raw[df_raw['model'] == model]['nmpjpe']
        clean = df_clean[df_clean['model'] == model]['nmpjpe']

        raw_mean = raw.mean()
        clean_mean = clean.mean()
        mean_diff = raw_mean - clean_mean

        raw_med = raw.median()
        clean_med = clean.median()
        med_diff = raw_med - clean_med

        print(f"| {model} | {raw_mean:.1f}% | {clean_mean:.1f}% | {mean_diff:+.1f}% | "
              f"{raw_med:.1f}% | {clean_med:.1f}% | {med_diff:+.1f}% |")

    # Per camera
    print("\n### By Camera (Mean)")
    print("\n| Model | Camera | Raw Mean | Clean Mean | Diff |")
    print("|-------|--------|----------|------------|------|")

    for model in df['model'].unique():
        for camera in ['c17', 'c18']:
            raw = df_raw[(df_raw['model'] == model) & (df_raw['camera'] == camera)]['nmpjpe']
            clean = df_clean[(df_clean['model'] == model) & (df_clean['camera'] == camera)]['nmpjpe']

            raw_mean = raw.mean()
            clean_mean = clean.mean()
            diff = raw_mean - clean_mean

            print(f"| {model} | {camera} | {raw_mean:.1f}% | {clean_mean:.1f}% | {diff:+.1f}% |")


# =============================================================================
# 4. EXTREME FRAME VERTEILUNG
# =============================================================================

def analyze_extreme_frame_distribution(df: pd.DataFrame, clean_only: bool = True):
    """Analyze where extreme frames occur."""
    if clean_only:
        df = df[~df['is_coach']]

    print("\n" + "=" * 70)
    print("VERTEILUNG DER >100% FRAMES")
    print("=" * 70)

    extreme = df[df['nmpjpe'] > 100]

    print(f"\nTotal >100% Frames: {len(extreme):,} ({len(extreme)/len(df)*100:.2f}% aller Frames)")

    # By model
    print("\n### Nach Modell")
    print("\n| Model | N Extreme | % of Model Frames |")
    print("|-------|-----------|-------------------|")

    for model in df['model'].unique():
        model_extreme = extreme[extreme['model'] == model]
        model_total = df[df['model'] == model]
        pct = len(model_extreme) / len(model_total) * 100
        print(f"| {model} | {len(model_extreme):,} | {pct:.2f}% |")

    # By camera
    print("\n### Nach Kamera")
    print("\n| Camera | N Extreme | % of Camera Frames |")
    print("|--------|-----------|-------------------|")

    for camera in ['c17', 'c18']:
        cam_extreme = extreme[extreme['camera'] == camera]
        cam_total = df[df['camera'] == camera]
        pct = len(cam_extreme) / len(cam_total) * 100
        print(f"| {camera} | {len(cam_extreme):,} | {pct:.2f}% |")

    # By exercise
    print("\n### Nach Übung")
    print("\n| Exercise | N Extreme | % of Exercise Frames |")
    print("|----------|-----------|---------------------|")

    for ex in sorted(df['exercise'].unique()):
        ex_extreme = extreme[extreme['exercise'] == ex]
        ex_total = df[df['exercise'] == ex]
        pct = len(ex_extreme) / len(ex_total) * 100
        print(f"| {ex} | {len(ex_extreme):,} | {pct:.2f}% |")

    # By rotation bucket
    print("\n### Nach Rotation")
    print("\n| Bucket | N Extreme | % of Bucket Frames |")
    print("|--------|-----------|-------------------|")

    df_with_rot = df[df['rotation_bucket'].notna()]
    extreme_with_rot = extreme[extreme['rotation_bucket'].notna()]

    for bucket in [f"{i}-{i+10}" for i in range(0, 90, 10)]:
        bucket_extreme = extreme_with_rot[extreme_with_rot['rotation_bucket'] == bucket]
        bucket_total = df_with_rot[df_with_rot['rotation_bucket'] == bucket]
        if len(bucket_total) == 0:
            continue
        pct = len(bucket_extreme) / len(bucket_total) * 100
        print(f"| {bucket} | {len(bucket_extreme):,} | {pct:.2f}% |")

    # Top problematic videos
    print("\n### Top 15 Problematische Videos (Clean, nicht Coach)")

    video_extreme = extreme.groupby('video_id').size().reset_index(name='n_extreme')
    video_total = df.groupby('video_id').size().reset_index(name='n_total')
    video_stats = video_extreme.merge(video_total, on='video_id')
    video_stats['pct'] = video_stats['n_extreme'] / video_stats['n_total'] * 100
    video_stats = video_stats.sort_values('n_extreme', ascending=False)

    print("\n| Video | N Extreme | Total Frames | % Extreme |")
    print("|-------|-----------|--------------|-----------|")

    for _, row in video_stats.head(15).iterrows():
        print(f"| {row['video_id']} | {row['n_extreme']:,} | {row['n_total']:,} | {row['pct']:.1f}% |")


# =============================================================================
# 5. SAUBERE MODELL-RANKINGS
# =============================================================================

def final_model_ranking(df: pd.DataFrame):
    """Final model ranking with multiple views."""
    df_clean = df[~df['is_coach']]
    df_no_extreme = df_clean[df_clean['nmpjpe'] <= 100]

    print("\n" + "=" * 70)
    print("FINALES MODELL-RANKING")
    print("=" * 70)

    print("\n### Szenario 1: Alle Clean-Daten (mit Ausreißern)")
    print("\n| Rank | Model | Mean | Median | Std |")
    print("|------|-------|------|--------|-----|")

    ranking1 = []
    for model in df_clean['model'].unique():
        m = df_clean[df_clean['model'] == model]['nmpjpe']
        ranking1.append((model, m.mean(), m.median(), m.std()))

    ranking1.sort(key=lambda x: x[2])  # Sort by median
    for i, (model, mean, median, std) in enumerate(ranking1, 1):
        print(f"| {i} | {model} | {mean:.1f}% | {median:.1f}% | {std:.1f}% |")

    print("\n### Szenario 2: Ohne >100% Frames (Person-Switch bereinigt)")
    print("\n| Rank | Model | Mean | Median | Std |")
    print("|------|-------|------|--------|-----|")

    ranking2 = []
    for model in df_no_extreme['model'].unique():
        m = df_no_extreme[df_no_extreme['model'] == model]['nmpjpe']
        ranking2.append((model, m.mean(), m.median(), m.std()))

    ranking2.sort(key=lambda x: x[2])  # Sort by median
    for i, (model, mean, median, std) in enumerate(ranking2, 1):
        print(f"| {i} | {model} | {mean:.1f}% | {median:.1f}% | {std:.1f}% |")

    print("\n### Interpretation")
    print("- Szenario 1 zeigt Robustheit gegenüber Multi-Person-Problemen")
    print("- Szenario 2 zeigt 'reine' Pose-Estimation-Genauigkeit")
    print("- Median ist robuster als Mean bei Ausreißern")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading frame-level data...")
    df = load_frame_data()
    print(f"Loaded {len(df):,} frames")

    # Run all analyses
    analyze_by_exercise(df, clean_only=True, filter_extreme=False)
    analyze_by_exercise(df, clean_only=True, filter_extreme=True)

    analyze_rotation_by_camera(df, clean_only=True, filter_extreme=False)
    analyze_rotation_by_camera(df, clean_only=True, filter_extreme=True)

    compare_with_without_extreme(df, clean_only=True)

    analyze_extreme_frame_distribution(df, clean_only=True)

    final_model_ranking(df)

    print("\n" + "=" * 70)
    print("EXTENDED ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
