"""
Generate all thesis-ready figures for the bachelor thesis.

Reads from existing analysis results (CSVs/JSONs) and produces
publication-quality figures with consistent styling.

Usage:
    .venv/Scripts/python analysis/create_thesis_figures.py

Output:
    analysis/thesis_figures/*.png and *.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("analysis/results")
OUTPUT_DIR = Path("analysis/thesis_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent color scheme across all figures
COLORS = {
    'MediaPipe': '#2ecc71',  # Green
    'MoveNet': '#3498db',    # Blue
    'YOLO': '#e74c3c',       # Red/Orange
}
MODEL_ORDER = ['MediaPipe', 'MoveNet', 'YOLO']
MODEL_LABELS = {
    'MediaPipe': 'MediaPipe',
    'MoveNet': 'MoveNet',
    'YOLO': 'YOLOv8-Pose',
}

# Plot style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
})


def save_figure(fig, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(OUTPUT_DIR / f"{name}.png")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    print(f"  Saved: {name}.png/.pdf")
    plt.close(fig)


# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")
df = pd.read_csv(RESULTS_DIR / "frame_level_data.csv")
jitter_df = pd.read_csv(RESULTS_DIR / "temporal_jitter.csv")
inference_df = pd.read_csv(RESULTS_DIR / "inference_benchmark.csv")
detection_df = pd.read_csv(RESULTS_DIR / "joint_detection_rate.csv")

# Clean data: no coach videos
df_clean = df[~df['is_coach']].copy()

# Clean + no person-switch outliers (>100% NMPJPE)
df_filtered = df_clean[df_clean['nmpjpe'] < 100].copy()

print(f"  Total frames: {len(df):,}")
print(f"  Clean frames: {len(df_clean):,}")
print(f"  Filtered frames (no outliers): {len(df_filtered):,}")


# ============================================================================
# Figure 1: Accuracy Comparison
# ============================================================================

def fig1_accuracy():
    """Bar chart comparing model accuracy with significance annotations."""
    print("\nFigure 1: Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(8, 5))

    stats = []
    for model in MODEL_ORDER:
        d = df_filtered[df_filtered['model'] == model]['nmpjpe']
        stats.append({
            'model': model,
            'mean': d.mean(),
            'median': d.median(),
            'std': d.std(),
        })

    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    means = [s['mean'] for s in stats]
    medians = [s['median'] for s in stats]
    stds = [s['std'] for s in stats]
    colors = [COLORS[m] for m in MODEL_ORDER]

    bars_mean = ax.bar(x - width/2, means, width, label='Mean',
                       color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    bars_median = ax.bar(x + width/2, medians, width, label='Median',
                         color=colors, alpha=0.5, edgecolor='white', linewidth=1.5,
                         hatch='///')

    # Add value labels
    for bar, val in zip(bars_mean, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars_median, medians):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Significance annotation: MediaPipe vs MoveNet (p=0.098, n.s.)
    y_sig = max(means) + 2.5
    ax.plot([x[0] - width/2, x[0] - width/2, x[1] - width/2, x[1] - width/2],
            [y_sig - 0.3, y_sig, y_sig, y_sig - 0.3], 'k-', linewidth=1)
    ax.text((x[0] + x[1])/2 - width/2, y_sig + 0.2, 'n.s. (p=0.098)',
            ha='center', fontsize=9, style='italic')

    ax.set_ylabel('NMPJPE (%)')
    ax.set_title('Model Accuracy on REHAB24-6 (Clean Data, Outliers Removed)')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(means) + 5)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure(fig, 'fig1_accuracy_comparison')


# ============================================================================
# Figure 2: Rotation Effect
# ============================================================================

def fig2_rotation():
    """Line plot showing NMPJPE vs rotation angle for each model."""
    print("\nFigure 2: Rotation Effect")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use c18 camera (lateral, has wider rotation range)
    df_rot = df_filtered[df_filtered['camera'] == 'c18'].copy()

    bucket_order = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']
    bucket_centers = [25, 35, 45, 55, 65, 75, 85]

    for model in MODEL_ORDER:
        model_data = df_rot[df_rot['model'] == model]
        medians = []
        q25s = []
        q75s = []
        for bucket in bucket_order:
            bucket_data = model_data[model_data['rotation_bucket'] == bucket]['nmpjpe']
            if len(bucket_data) > 0:
                medians.append(bucket_data.median())
                q25s.append(bucket_data.quantile(0.25))
                q75s.append(bucket_data.quantile(0.75))
            else:
                medians.append(np.nan)
                q25s.append(np.nan)
                q75s.append(np.nan)

        ax.plot(bucket_centers, medians, 'o-', color=COLORS[model],
                label=MODEL_LABELS[model], linewidth=2, markersize=6)
        ax.fill_between(bucket_centers, q25s, q75s, color=COLORS[model], alpha=0.15)

    # Annotate regions
    ax.axvspan(20, 40, alpha=0.05, color='green', label='_nolegend_')
    ax.axvspan(70, 90, alpha=0.05, color='red', label='_nolegend_')
    ax.text(30, ax.get_ylim()[1] * 0.95, 'Frontal/Diagonal', ha='center',
            fontsize=9, color='green', alpha=0.7)
    ax.text(80, ax.get_ylim()[1] * 0.95, 'Lateral', ha='center',
            fontsize=9, color='red', alpha=0.7)

    ax.set_xlabel('Rotation Angle (degrees)')
    ax.set_ylabel('Median NMPJPE (%)')
    ax.set_title('Rotation Effect on Pose Estimation Accuracy (c18 Camera)')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure(fig, 'fig2_rotation_effect')


# ============================================================================
# Figure 3: Multi-Person Robustness
# ============================================================================

def fig3_multiperson():
    """Grouped bar chart: Clean vs Coach performance."""
    print("\nFigure 3: Multi-Person Robustness")
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(MODEL_ORDER))
    width = 0.3

    clean_means = []
    coach_means = []
    for model in MODEL_ORDER:
        clean = df[~df['is_coach'] & (df['model'] == model)]['nmpjpe']
        coach = df[df['is_coach'] & (df['model'] == model)]['nmpjpe']
        clean_means.append(clean.mean())
        coach_means.append(coach.mean())

    bars1 = ax.bar(x - width/2, clean_means, width, label='Clean (121 videos)',
                   color=[COLORS[m] for m in MODEL_ORDER], alpha=0.85,
                   edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, coach_means, width, label='With Coach (5 videos)',
                   color=[COLORS[m] for m in MODEL_ORDER], alpha=0.4,
                   edgecolor=[COLORS[m] for m in MODEL_ORDER], linewidth=1.5,
                   hatch='xxx')

    # Add value labels and increase percentages
    for i, (c, co) in enumerate(zip(clean_means, coach_means)):
        ax.text(x[i] - width/2, c + 1, f'{c:.1f}%', ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i] + width/2, co + 1, f'{co:.1f}%', ha='center', fontsize=9)
        increase = (co - c) / c * 100
        ax.annotate(f'+{increase:.0f}%', xy=(x[i] + width/2, co + 4),
                   fontsize=10, fontweight='bold', ha='center', color='#c0392b')

    # Combine model names with selection strategy in x-tick labels
    strategies = ['Torso Size', 'BBox Area', 'BBox Area']
    combined_labels = [f'{MODEL_LABELS[m]}\n({strat})' for m, strat in zip(MODEL_ORDER, strategies)]

    ax.set_ylabel('Mean NMPJPE (%)')
    ax.set_title('Multi-Person Robustness: Selection Strategy Impact')
    ax.set_xticks(x)
    ax.set_xticklabels(combined_labels)
    ax.legend(loc='upper left')
    ax.set_ylim(0, max(coach_means) + 12)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure(fig, 'fig3_multiperson_robustness')


# ============================================================================
# Figure 4: Temporal Stability (Jitter)
# ============================================================================

def fig4_jitter():
    """Bar chart comparing frame-to-frame jitter."""
    print("\nFigure 4: Temporal Stability")
    fig, ax = plt.subplots(figsize=(8, 5))

    jitter_data = {}
    for _, row in jitter_df.iterrows():
        model = row['model']
        jitter_data[model] = {
            'mean': row['mean_jitter'],
            'median': row['median_jitter'],
            'p90': row['p90_jitter'],
        }

    x = np.arange(len(MODEL_ORDER))
    width = 0.25

    means = [jitter_data[m]['mean'] for m in MODEL_ORDER]
    medians = [jitter_data[m]['median'] for m in MODEL_ORDER]
    p90s = [jitter_data[m]['p90'] for m in MODEL_ORDER]
    colors = [COLORS[m] for m in MODEL_ORDER]

    bars1 = ax.bar(x - width, means, width, label='Mean', color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x, medians, width, label='Median', color=colors, alpha=0.5,
                   edgecolor='white', linewidth=1.5, hatch='///')
    bars3 = ax.bar(x + width, p90s, width, label='P90', color=colors, alpha=0.35,
                   edgecolor='white', linewidth=1.5, hatch='...')

    for bar, val in zip(bars1, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}%', ha='center', fontsize=9, fontweight='bold')

    # Highlight winner
    ax.annotate('42% less\njitter', xy=(x[1] - width, means[1]),
               xytext=(x[1] - width - 0.5, means[1] + 0.8),
               fontsize=9, color=COLORS['MoveNet'], fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=COLORS['MoveNet']))

    ax.set_ylabel('Jitter (% of torso length)')
    ax.set_title('Temporal Stability: Frame-to-Frame Prediction Consistency')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure(fig, 'fig4_temporal_stability')


# ============================================================================
# Figure 5: Inference Time
# ============================================================================

def fig5_inference():
    """Bar chart comparing inference speed (FPS)."""
    print("\nFigure 5: Inference Time")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    model_map = {
        'MediaPipe (full)': 'MediaPipe',
        'MoveNet (multipose)': 'MoveNet',
        'YOLOv8-Pose (n)': 'YOLO',
    }

    inf_data = {}
    for _, row in inference_df.iterrows():
        model = model_map.get(row['model'], row['model'])
        inf_data[model] = {
            'fps': row['fps'],
            'mean_ms': row['mean_ms'],
            'median_ms': row['median_ms'],
            'p95_ms': row['p95_ms'],
            'std_ms': row['std_ms'],
        }

    x = np.arange(len(MODEL_ORDER))
    colors = [COLORS[m] for m in MODEL_ORDER]

    # Left: FPS
    fps_vals = [inf_data[m]['fps'] for m in MODEL_ORDER]
    bars = ax1.bar(x, fps_vals, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, fps_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
    ax1.axhline(y=30, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(2.4, 30.5, '30 FPS\n(real-time)', fontsize=8, color='gray', ha='right')
    ax1.set_ylabel('Frames per Second (FPS)')
    ax1.set_title('Inference Speed')
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax1.set_ylim(0, max(fps_vals) + 5)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: Latency distribution
    mean_vals = [inf_data[m]['mean_ms'] for m in MODEL_ORDER]
    p95_vals = [inf_data[m]['p95_ms'] for m in MODEL_ORDER]
    std_vals = [inf_data[m]['std_ms'] for m in MODEL_ORDER]

    bars = ax2.bar(x, mean_vals, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax2.errorbar(x, mean_vals, yerr=std_vals, fmt='none', ecolor='black',
                 capsize=5, capthick=1.5, linewidth=1.5)

    # P95 markers
    ax2.scatter(x, p95_vals, marker='v', color='black', s=60, zorder=5, label='P95')

    for bar, val in zip(bars, mean_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_vals[MODEL_ORDER.index(MODEL_ORDER[int(bar.get_x() + 0.5)])] + 2,
                f'{val:.1f}ms', ha='center', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Inference Latency (CPU only)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Inference Time Benchmark (AMD Ryzen CPU, 16GB RAM, No GPU)', fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, 'fig5_inference_time')


# ============================================================================
# Figure 6: Per-Joint Heatmap
# ============================================================================

def fig6_perjoint():
    """Heatmap showing error by joint and model."""
    print("\nFigure 6: Per-Joint Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))

    joint_cols = [
        'error_left_shoulder', 'error_right_shoulder',
        'error_left_elbow', 'error_right_elbow',
        'error_left_wrist', 'error_right_wrist',
        'error_left_hip', 'error_right_hip',
        'error_left_knee', 'error_right_knee',
        'error_left_ankle', 'error_right_ankle',
    ]
    joint_labels = [
        'L Shoulder', 'R Shoulder',
        'L Elbow', 'R Elbow',
        'L Wrist', 'R Wrist',
        'L Hip', 'R Hip',
        'L Knee', 'R Knee',
        'L Ankle', 'R Ankle',
    ]

    heatmap_data = np.zeros((len(MODEL_ORDER), len(joint_cols)))

    for i, model in enumerate(MODEL_ORDER):
        model_data = df_filtered[df_filtered['model'] == model]
        for j, col in enumerate(joint_cols):
            valid = model_data[col].replace(0, np.nan).dropna()
            if len(valid) > 0:
                heatmap_data[i, j] = valid.median()

    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=5, vmax=18)

    # Add text annotations
    for i in range(len(MODEL_ORDER)):
        for j in range(len(joint_cols)):
            val = heatmap_data[i, j]
            color = 'white' if val > 14 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    ax.set_xticks(range(len(joint_labels)))
    ax.set_xticklabels(joint_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_title('Per-Joint Median Error (NMPJPE %)')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Median NMPJPE (%)')

    fig.tight_layout()
    save_figure(fig, 'fig6_perjoint_heatmap')


# ============================================================================
# Figure 7: Detection Completeness
# ============================================================================

def fig7_detection():
    """Bar chart showing detection completeness."""
    print("\nFigure 7: Detection Completeness")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Overall completeness (% frames with 12/12 joints)
    x = np.arange(len(MODEL_ORDER))
    completeness = []
    for model in MODEL_ORDER:
        model_data = df_clean[df_clean['model'] == model]
        full_detection = (model_data['valid_joints'] == 12).sum() / len(model_data) * 100
        completeness.append(full_detection)

    colors = [COLORS[m] for m in MODEL_ORDER]
    bars = ax1.bar(x, completeness, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, completeness):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Frames with Complete Detection (%)')
    ax1.set_title('Detection Completeness (12/12 Joints)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: Per-joint detection rate (problematic joints only)
    problem_joints = ['Right Elbow', 'Right Wrist', 'Right Knee', 'Right Ankle',
                      'Left Elbow', 'Left Wrist']
    joint_x = np.arange(len(problem_joints))
    width = 0.25

    for i, model in enumerate(MODEL_ORDER):
        model_det = detection_df[detection_df['model'] == model]
        rates = []
        for joint in problem_joints:
            row = model_det[model_det['joint'] == joint]
            rates.append(row['detection_rate'].values[0] if len(row) > 0 else 0)
        ax2.bar(joint_x + i * width, rates, width, color=COLORS[model],
                alpha=0.85, label=MODEL_LABELS[model], edgecolor='white', linewidth=1)

    ax2.set_ylabel('Detection Rate (%)')
    ax2.set_title('Per-Joint Detection Rate (Problematic Joints)')
    ax2.set_xticks(joint_x + width)
    ax2.set_xticklabels(problem_joints, rotation=35, ha='right')
    ax2.set_ylim(70, 101)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    save_figure(fig, 'fig7_detection_completeness')


# ============================================================================
# Figure 8: Trade-off Radar Chart (THE Key Figure)
# ============================================================================

def fig8_radar():
    """6-axis radar chart comparing all models across all dimensions."""
    print("\nFigure 8: Trade-off Radar Chart")

    categories = [
        'Accuracy',
        'Speed (FPS)',
        'Rotation\nRobustness',
        'Multi-Person\nRobustness',
        'Temporal\nStability',
        'Detection\nCompleteness',
    ]
    N = len(categories)

    # Compute normalized scores (0-100, higher = better)
    model_map_inf = {
        'MediaPipe (full)': 'MediaPipe',
        'MoveNet (multipose)': 'MoveNet',
        'YOLOv8-Pose (n)': 'YOLO',
    }

    # Raw values
    # Accuracy: median NMPJPE (lower is better)
    acc = {}
    for m in MODEL_ORDER:
        acc[m] = df_filtered[df_filtered['model'] == m]['nmpjpe'].median()

    # Speed: FPS (higher is better)
    fps = {}
    for _, row in inference_df.iterrows():
        model = model_map_inf.get(row['model'])
        if model:
            fps[model] = row['fps']

    # Rotation robustness: relative increase frontal->lateral (lower is better)
    rot_increase = {}
    for m in MODEL_ORDER:
        d = df_filtered[(df_filtered['model'] == m) & (df_filtered['camera'] == 'c18')]
        frontal = d[d['rotation_bucket'].isin(['20-30', '30-40'])]['nmpjpe'].median()
        lateral = d[d['rotation_bucket'].isin(['80-90'])]['nmpjpe'].median()
        rot_increase[m] = (lateral - frontal) / frontal * 100 if frontal > 0 else 0

    # Multi-person robustness: coach increase % (lower is better)
    mp_increase = {}
    for m in MODEL_ORDER:
        clean = df[~df['is_coach'] & (df['model'] == m)]['nmpjpe'].mean()
        coach = df[df['is_coach'] & (df['model'] == m)]['nmpjpe'].mean()
        mp_increase[m] = (coach - clean) / clean * 100

    # Temporal stability: mean jitter (lower is better)
    jitter = {}
    for _, row in jitter_df.iterrows():
        jitter[row['model']] = row['mean_jitter']

    # Detection completeness: % frames with 12/12 joints (higher is better)
    det_rate = {}
    for m in MODEL_ORDER:
        model_data = df_clean[df_clean['model'] == m]
        det_rate[m] = (model_data['valid_joints'] == 12).sum() / len(model_data) * 100

    # Normalize to 0-100 scale (higher = better for all)
    def normalize_lower_better(vals, worst_factor=1.3):
        """Normalize where lower raw value = higher score."""
        worst = max(vals.values()) * worst_factor
        return {k: max(0, (1 - v / worst)) * 100 for k, v in vals.items()}

    def normalize_higher_better(vals, best_factor=1.1):
        """Normalize where higher raw value = higher score."""
        best = max(vals.values()) * best_factor
        return {k: (v / best) * 100 for k, v in vals.items()}

    scores = {}
    for m in MODEL_ORDER:
        scores[m] = [
            normalize_lower_better(acc)[m],
            normalize_higher_better(fps)[m],
            normalize_lower_better(rot_increase)[m],
            normalize_lower_better(mp_increase)[m],
            normalize_lower_better(jitter)[m],
            normalize_higher_better(det_rate)[m],
        ]

    # Plot
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in MODEL_ORDER:
        values = scores[model] + scores[model][:1]
        ax.plot(angles, values, 'o-', color=COLORS[model],
                label=MODEL_LABELS[model], linewidth=2.5, markersize=6)
        ax.fill(angles, values, color=COLORS[model], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8, color='gray')
    ax.set_title('Multi-Criteria Model Comparison\n(Higher is Better)', fontsize=14, pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.2, -0.05), fontsize=11)

    save_figure(fig, 'fig8_tradeoff_radar')


# ============================================================================
# Figure 9: Qualitative Example (Coach Tracking)
# ============================================================================

def fig9_qualitative():
    """Show frame where coach gets tracked instead of patient."""
    print("\nFigure 9: Qualitative Example")

    pred_path = Path("data/predictions/Ex3/PM_119-c17.npz")
    gt_path = Path("data/gt_2d/Ex3/PM_119-c17-30fps.npy")
    video_path = Path("data/videos/Ex3/PM_119-Camera17-30fps.mp4")

    if not pred_path.exists() or not gt_path.exists() or not video_path.exists():
        print("  SKIPPED: Required files not found")
        print(f"    Predictions: {pred_path.exists()}")
        print(f"    Ground Truth: {gt_path.exists()}")
        print(f"    Video: {video_path.exists()}")
        return

    try:
        import cv2
    except ImportError:
        print("  SKIPPED: OpenCV not available")
        return

    # Load data
    preds = np.load(pred_path)
    gt_2d = np.load(gt_path)

    # GT joint mapping (GT index -> COCO index for 12 comparable joints)
    GT_TO_COCO = {7: 5, 12: 6, 8: 7, 13: 8, 9: 9, 14: 10, 16: 11, 21: 12, 17: 13, 22: 14, 18: 15, 23: 16}
    COCO_TO_GT = {v: k for k, v in GT_TO_COCO.items()}
    COMPARABLE_COCO = sorted(GT_TO_COCO.values())

    # Get predictions for all models
    model_keys = {
        'MediaPipe': [k for k in preds.files if 'MediaPipe' in k][0],
        'MoveNet': [k for k in preds.files if 'MoveNet' in k][0],
        'YOLO': [k for k in preds.files if 'YOLO' in k or 'yolo' in k.lower()][0],
    }

    # Find a frame with high error (person switch) - check MoveNet predictions
    pred_movenet = preds[model_keys['MoveNet']]  # (N, 17, 3)

    # frame_step=3 was used, so prediction frame i = video frame i*3
    # Compute per-frame error for MoveNet to find worst frame
    num_pred_frames = pred_movenet.shape[0]
    num_gt_frames = gt_2d.shape[0]
    frame_step = max(1, num_gt_frames // num_pred_frames)

    errors = []
    for i in range(min(num_pred_frames, num_gt_frames // frame_step)):
        gt_frame_idx = i * frame_step
        if gt_frame_idx >= num_gt_frames:
            break

        gt_kps = np.array([gt_2d[gt_frame_idx, COCO_TO_GT[c]] for c in COMPARABLE_COCO])
        pred_kps = np.array([pred_movenet[i, c, :2] for c in COMPARABLE_COCO])

        # Torso length for normalization
        l_shoulder_gt = gt_2d[gt_frame_idx, 7]
        r_shoulder_gt = gt_2d[gt_frame_idx, 12]
        l_hip_gt = gt_2d[gt_frame_idx, 16]
        r_hip_gt = gt_2d[gt_frame_idx, 21]
        shoulder_mid = (l_shoulder_gt + r_shoulder_gt) / 2
        hip_mid = (l_hip_gt + r_hip_gt) / 2
        torso = np.linalg.norm(shoulder_mid - hip_mid)
        if torso < 10:
            errors.append(0)
            continue

        dists = np.linalg.norm(pred_kps - gt_kps, axis=1)
        nmpjpe = np.mean(dists) / torso * 100
        errors.append(nmpjpe)

    errors = np.array(errors)

    # Also compute MediaPipe errors to find a frame where MediaPipe is OK but MoveNet fails
    pred_mediapipe = preds[model_keys['MediaPipe']]
    mp_errors = []
    for i in range(min(num_pred_frames, num_gt_frames // frame_step)):
        gt_frame_idx = i * frame_step
        if gt_frame_idx >= num_gt_frames:
            break
        gt_kps = np.array([gt_2d[gt_frame_idx, COCO_TO_GT[c]] for c in COMPARABLE_COCO])
        pred_kps = np.array([pred_mediapipe[i, c, :2] for c in COMPARABLE_COCO])
        l_shoulder_gt = gt_2d[gt_frame_idx, 7]
        r_shoulder_gt = gt_2d[gt_frame_idx, 12]
        l_hip_gt = gt_2d[gt_frame_idx, 16]
        r_hip_gt = gt_2d[gt_frame_idx, 21]
        shoulder_mid = (l_shoulder_gt + r_shoulder_gt) / 2
        hip_mid = (l_hip_gt + r_hip_gt) / 2
        torso = np.linalg.norm(shoulder_mid - hip_mid)
        if torso < 10:
            mp_errors.append(0)
            continue
        dists = np.linalg.norm(pred_kps - gt_kps, axis=1)
        mp_errors.append(np.mean(dists) / torso * 100)
    mp_errors = np.array(mp_errors)

    # Best case: MediaPipe correct (<30%) but MoveNet wrong (>80%)
    # This shows the selection strategy difference
    selection_diff = np.where((mp_errors < 30) & (errors > 80))[0]
    if len(selection_diff) > 0:
        target_pred_idx = selection_diff[len(selection_diff) // 2]
        print(f"  Found selection-difference frame! MediaPipe={mp_errors[target_pred_idx]:.1f}%, MoveNet={errors[target_pred_idx]:.1f}%")
    else:
        # Fallback: frame where all models fail
        high_error_frames = np.where(errors > 80)[0]
        if len(high_error_frames) == 0:
            high_error_frames = np.where(errors > 50)[0]
        if len(high_error_frames) == 0:
            high_error_frames = [np.argmax(errors)]
        target_pred_idx = high_error_frames[len(high_error_frames) // 2]
        print(f"  No selection-diff frame found, using all-fail frame")

    target_video_frame = target_pred_idx * frame_step

    print(f"  Selected frame: pred_idx={target_pred_idx}, video_frame={target_video_frame}")
    print(f"  MoveNet error at this frame: {errors[target_pred_idx]:.1f}%")
    print(f"  MediaPipe error at this frame: {mp_errors[target_pred_idx]:.1f}%")

    # Extract video frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_video_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("  SKIPPED: Could not read video frame")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create figure with 3 subplots (one per model)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    gt_frame_idx = target_video_frame
    gt_kps = np.array([gt_2d[gt_frame_idx, COCO_TO_GT[c]] for c in COMPARABLE_COCO])

    strategies_map = {
        'MediaPipe': 'Torso Size',
        'MoveNet': 'BBox Area',
        'YOLO': 'BBox Area',
    }

    # Skeleton connections (COCO indices for 12 comparable)
    # Map to 0-11 index: [5,6,7,8,9,10,11,12,13,14,15,16] -> [0,1,2,3,4,5,6,7,8,9,10,11]
    skeleton = [
        (0, 1),   # L shoulder - R shoulder
        (0, 2),   # L shoulder - L elbow
        (1, 3),   # R shoulder - R elbow
        (2, 4),   # L elbow - L wrist
        (3, 5),   # R elbow - R wrist
        (0, 6),   # L shoulder - L hip
        (1, 7),   # R shoulder - R hip
        (6, 7),   # L hip - R hip
        (6, 8),   # L hip - L knee
        (7, 9),   # R hip - R knee
        (8, 10),  # L knee - L ankle
        (9, 11),  # R knee - R ankle
    ]

    for ax_idx, model in enumerate(MODEL_ORDER):
        ax = axes[ax_idx]
        ax.imshow(frame_rgb)

        pred_data = preds[model_keys[model]]
        pred_kps = np.array([pred_data[target_pred_idx, c, :2] for c in COMPARABLE_COCO])
        pred_conf = np.array([pred_data[target_pred_idx, c, 2] for c in COMPARABLE_COCO])

        # Compute error for this model
        l_shoulder_gt = gt_2d[gt_frame_idx, 7]
        r_shoulder_gt = gt_2d[gt_frame_idx, 12]
        l_hip_gt = gt_2d[gt_frame_idx, 16]
        r_hip_gt = gt_2d[gt_frame_idx, 21]
        shoulder_mid = (l_shoulder_gt + r_shoulder_gt) / 2
        hip_mid = (l_hip_gt + r_hip_gt) / 2
        torso = np.linalg.norm(shoulder_mid - hip_mid)
        dists = np.linalg.norm(pred_kps - gt_kps, axis=1)
        nmpjpe = np.mean(dists) / torso * 100 if torso > 10 else 0

        # Draw GT skeleton (green)
        for i, j in skeleton:
            if gt_kps[i][0] > 0 and gt_kps[j][0] > 0:
                ax.plot([gt_kps[i][0], gt_kps[j][0]],
                       [gt_kps[i][1], gt_kps[j][1]],
                       'g-', linewidth=2, alpha=0.7)
        ax.scatter(gt_kps[:, 0], gt_kps[:, 1], c='green', s=40, zorder=5,
                  edgecolors='white', linewidth=1)

        # Draw prediction skeleton (red/model color)
        valid = pred_conf > 0.1
        for i, j in skeleton:
            if valid[i] and valid[j] and pred_kps[i][0] > 0 and pred_kps[j][0] > 0:
                ax.plot([pred_kps[i][0], pred_kps[j][0]],
                       [pred_kps[i][1], pred_kps[j][1]],
                       '-', color=COLORS[model], linewidth=2, alpha=0.8)
        valid_kps = pred_kps[valid]
        ax.scatter(valid_kps[:, 0], valid_kps[:, 1], c=COLORS[model], s=40,
                  zorder=5, edgecolors='white', linewidth=1)

        title_color = '#27ae60' if nmpjpe < 30 else '#c0392b'
        status = 'CORRECT' if nmpjpe < 30 else 'WRONG PERSON'
        ax.set_title(f'{MODEL_LABELS[model]} ({strategies_map[model]})\n'
                    f'NMPJPE: {nmpjpe:.1f}% - {status}',
                    fontsize=12, fontweight='bold', color=title_color)
        ax.axis('off')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='green', label='Ground Truth (Patient)',
               markersize=8, linestyle='-', linewidth=2),
        Line2D([0], [0], marker='o', color='gray', label='Prediction (tracked person)',
               markersize=8, linestyle='-', linewidth=2),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
              fontsize=11, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'Person Selection Strategy: Torso Size vs BBox Area\n'
                f'PM_119-c17, Frame {target_video_frame}',
                fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, 'fig9_qualitative_coach')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Generating Thesis Figures")
    print("=" * 60)

    fig1_accuracy()
    fig2_rotation()
    fig3_multiperson()
    fig4_jitter()
    fig5_inference()
    fig6_perjoint()
    fig7_detection()
    fig8_radar()
    fig9_qualitative()

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}/")
    print("=" * 60)
