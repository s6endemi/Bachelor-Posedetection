"""
Generate publication-ready figures for the evaluation results.

Uses the frame_level_data.csv from comprehensive_analysis.py.

Author: Eren
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Colors
COLORS = {
    'MediaPipe': '#4CAF50',  # Green
    'MoveNet': '#2196F3',    # Blue
    'YOLO': '#FF9800'        # Orange
}


def load_data():
    """Load frame-level data."""
    df = pd.read_csv(RESULTS_DIR / "frame_level_data.csv")
    return df


# =============================================================================
# FIGURE 1: Model Comparison Boxplot
# =============================================================================

def fig1_model_comparison(df):
    """Boxplot comparing NMPJPE distribution across models."""

    # Clean data only, cap at 50% for visibility
    clean = df[~df['is_coach']].copy()
    clean['nmpjpe_capped'] = clean['nmpjpe'].clip(upper=50)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Raw data
    ax1 = axes[0]
    data_raw = [clean[clean['model'] == m]['nmpjpe_capped'].values for m in ['MediaPipe', 'MoveNet', 'YOLO']]
    bp1 = ax1.boxplot(data_raw, labels=['MediaPipe', 'MoveNet', 'YOLO'], patch_artist=True)

    for patch, model in zip(bp1['boxes'], ['MediaPipe', 'MoveNet', 'YOLO']):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.7)

    ax1.set_ylabel('NMPJPE (%)')
    ax1.set_title('A) All Clean Data')
    ax1.set_ylim(0, 55)

    # Add median annotations
    for i, model in enumerate(['MediaPipe', 'MoveNet', 'YOLO'], 1):
        median = clean[clean['model'] == model]['nmpjpe'].median()
        ax1.annotate(f'Median: {median:.1f}%', xy=(i, median), xytext=(i+0.3, median+2),
                    fontsize=9, ha='left')

    # Right: Without extreme frames
    ax2 = axes[1]
    clean_no_extreme = clean[clean['nmpjpe'] <= 100]
    data_clean = [clean_no_extreme[clean_no_extreme['model'] == m]['nmpjpe'].values for m in ['MediaPipe', 'MoveNet', 'YOLO']]
    bp2 = ax2.boxplot(data_clean, labels=['MediaPipe', 'MoveNet', 'YOLO'], patch_artist=True)

    for patch, model in zip(bp2['boxes'], ['MediaPipe', 'MoveNet', 'YOLO']):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.7)

    ax2.set_ylabel('NMPJPE (%)')
    ax2.set_title('B) Without Person-Switch Frames (>100%)')
    ax2.set_ylim(0, 35)

    # Add median annotations
    for i, model in enumerate(['MediaPipe', 'MoveNet', 'YOLO'], 1):
        median = clean_no_extreme[clean_no_extreme['model'] == model]['nmpjpe'].median()
        ax2.annotate(f'Median: {median:.1f}%', xy=(i, median), xytext=(i+0.3, median+1),
                    fontsize=9, ha='left')

    plt.tight_layout()

    # Save
    fig.savefig(FIGURES_DIR / 'fig1_model_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig1_model_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("Created fig1_model_comparison")


# =============================================================================
# FIGURE 2: Rotation Effect
# =============================================================================

def fig2_rotation_effect(df):
    """Line plot showing NMPJPE vs rotation angle."""

    # c18 only, clean, no extreme
    clean = df[(~df['is_coach']) & (df['camera'] == 'c18') & (df['nmpjpe'] <= 100)]
    clean = clean[clean['rotation_bucket'].notna()]

    fig, ax = plt.subplots(figsize=(10, 6))

    buckets = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']
    x_positions = [25, 35, 45, 55, 65, 75, 85]

    for model in ['MediaPipe', 'MoveNet', 'YOLO']:
        model_df = clean[clean['model'] == model]

        medians = []
        p25s = []
        p75s = []

        for bucket in buckets:
            bucket_df = model_df[model_df['rotation_bucket'] == bucket]
            if len(bucket_df) > 0:
                medians.append(bucket_df['nmpjpe'].median())
                p25s.append(bucket_df['nmpjpe'].quantile(0.25))
                p75s.append(bucket_df['nmpjpe'].quantile(0.75))
            else:
                medians.append(np.nan)
                p25s.append(np.nan)
                p75s.append(np.nan)

        # Plot line
        ax.plot(x_positions, medians, 'o-', color=COLORS[model], label=model, linewidth=2, markersize=8)

        # Plot confidence band
        ax.fill_between(x_positions, p25s, p75s, color=COLORS[model], alpha=0.2)

    ax.set_xlabel('Rotation Angle (degrees)')
    ax.set_ylabel('Median NMPJPE (%)')
    ax.set_title('Rotation Effect on Pose Estimation Accuracy (c18 Camera)')
    ax.legend(loc='upper left')
    ax.set_xlim(20, 90)
    ax.set_ylim(8, 18)
    ax.set_xticks([30, 40, 50, 60, 70, 80, 90])
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='10% baseline')

    # Add annotations
    ax.annotate('Frontal/Diagonal', xy=(30, 17.5), fontsize=10, ha='center', color='gray')
    ax.annotate('Lateral', xy=(80, 17.5), fontsize=10, ha='center', color='gray')

    plt.tight_layout()

    fig.savefig(FIGURES_DIR / 'fig2_rotation_effect.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig2_rotation_effect.pdf', bbox_inches='tight')
    plt.close()
    print("Created fig2_rotation_effect")


# =============================================================================
# FIGURE 3: Camera Comparison
# =============================================================================

def fig3_camera_comparison(df):
    """Bar chart comparing c17 vs c18 performance."""

    clean = df[~df['is_coach']]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = ['MediaPipe', 'MoveNet', 'YOLO']
    x = np.arange(len(models))
    width = 0.35

    # Left: Mean NMPJPE
    ax1 = axes[0]

    c17_means = [clean[(clean['model'] == m) & (clean['camera'] == 'c17')]['nmpjpe'].mean() for m in models]
    c18_means = [clean[(clean['model'] == m) & (clean['camera'] == 'c18')]['nmpjpe'].mean() for m in models]

    bars1 = ax1.bar(x - width/2, c17_means, width, label='c17 (frontal)', color='#1976D2')
    bars2 = ax1.bar(x + width/2, c18_means, width, label='c18 (lateral)', color='#FF7043')

    ax1.set_ylabel('Mean NMPJPE (%)')
    ax1.set_title('A) Raw Mean (with outliers)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim(0, 25)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    # Right: Without extreme frames
    ax2 = axes[1]

    clean_no_extreme = clean[clean['nmpjpe'] <= 100]

    c17_means_clean = [clean_no_extreme[(clean_no_extreme['model'] == m) & (clean_no_extreme['camera'] == 'c17')]['nmpjpe'].mean() for m in models]
    c18_means_clean = [clean_no_extreme[(clean_no_extreme['model'] == m) & (clean_no_extreme['camera'] == 'c18')]['nmpjpe'].mean() for m in models]

    bars3 = ax2.bar(x - width/2, c17_means_clean, width, label='c17 (frontal)', color='#1976D2')
    bars4 = ax2.bar(x + width/2, c18_means_clean, width, label='c18 (lateral)', color='#FF7043')

    ax2.set_ylabel('Mean NMPJPE (%)')
    ax2.set_title('B) Cleaned Mean (no >100% frames)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.set_ylim(0, 18)

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()

    fig.savefig(FIGURES_DIR / 'fig3_camera_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig3_camera_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("Created fig3_camera_comparison")


# =============================================================================
# FIGURE 4: Selection Robustness
# =============================================================================

def fig4_selection_robustness(df):
    """Bar chart showing clean vs coach performance."""

    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['MediaPipe', 'MoveNet', 'YOLO']
    x = np.arange(len(models))
    width = 0.35

    clean_means = [df[(df['model'] == m) & (~df['is_coach'])]['nmpjpe'].mean() for m in models]
    coach_means = [df[(df['model'] == m) & (df['is_coach'])]['nmpjpe'].mean() for m in models]

    bars1 = ax.bar(x - width/2, clean_means, width, label='Clean (no coach)', color='#4CAF50')
    bars2 = ax.bar(x + width/2, coach_means, width, label='With Coach', color='#F44336')

    ax.set_ylabel('Mean NMPJPE (%)')
    ax.set_title('Selection Robustness: Impact of Coach Presence')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 75)

    # Add value labels and increase percentages
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        h1 = bar1.get_height()
        h2 = bar2.get_height()
        increase = (h2 - h1) / h1 * 100

        ax.annotate(f'{h1:.1f}%', xy=(bar1.get_x() + bar1.get_width()/2, h1),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        ax.annotate(f'{h2:.1f}%', xy=(bar2.get_x() + bar2.get_width()/2, h2),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

        # Add increase annotation
        ax.annotate(f'+{increase:.0f}%', xy=(x[i], (h1 + h2) / 2 + 5),
                    fontsize=10, ha='center', color='#D32F2F', fontweight='bold')

    # Add annotation for MediaPipe
    ax.annotate('Most robust\n(Torso-Selection)', xy=(0, 50), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50'))

    plt.tight_layout()

    fig.savefig(FIGURES_DIR / 'fig4_selection_robustness.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig4_selection_robustness.pdf', bbox_inches='tight')
    plt.close()
    print("Created fig4_selection_robustness")


# =============================================================================
# FIGURE 5: Extreme Frames Distribution
# =============================================================================

def fig5_extreme_distribution(df):
    """Pie/bar showing where extreme frames occur."""

    clean = df[~df['is_coach']]
    extreme = clean[clean['nmpjpe'] > 100]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: By camera
    ax1 = axes[0]

    c17_extreme = len(extreme[extreme['camera'] == 'c17'])
    c18_extreme = len(extreme[extreme['camera'] == 'c18'])

    bars = ax1.bar(['c17 (frontal)', 'c18 (lateral)'], [c17_extreme, c18_extreme],
                   color=['#1976D2', '#FF7043'])

    ax1.set_ylabel('Number of >100% Frames')
    ax1.set_title('A) Extreme Frames by Camera')

    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{int(height):,}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11)

    # Add ratio annotation
    ratio = c17_extreme / c18_extreme if c18_extreme > 0 else 0
    ax1.annotate(f'c17 has {ratio:.0f}x more\nextreme frames', xy=(0.5, max(c17_extreme, c18_extreme) * 0.7),
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Right: By model
    ax2 = axes[1]

    models = ['MediaPipe', 'MoveNet', 'YOLO']
    model_extreme = [len(extreme[extreme['model'] == m]) for m in models]

    bars2 = ax2.bar(models, model_extreme, color=[COLORS[m] for m in models])

    ax2.set_ylabel('Number of >100% Frames')
    ax2.set_title('B) Extreme Frames by Model')

    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{int(height):,}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11)

    plt.tight_layout()

    fig.savefig(FIGURES_DIR / 'fig5_extreme_distribution.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig5_extreme_distribution.pdf', bbox_inches='tight')
    plt.close()
    print("Created fig5_extreme_distribution")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} frames")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")
    fig1_model_comparison(df)
    fig2_rotation_effect(df)
    fig3_camera_comparison(df)
    fig4_selection_robustness(df)
    fig5_extreme_distribution(df)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
