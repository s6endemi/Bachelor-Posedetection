"""
Publication-Quality Visualizations for Bachelor Thesis
Pose Estimation Comparison on REHAB24-6 Dataset

Creates figures for:
1. Model Benchmark (Bar Chart with Error Bars)
2. Selection Robustness (Clean vs Coach Comparison)
3. Camera Perspective Effect
4. Per-Joint Analysis Heatmap
5. Confidence Calibration

Output: analysis/figures/
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif'
})

# Color palette - distinct colors for each model
COLORS = {
    'MediaPipe': '#2ecc71',  # Green
    'MoveNet': '#3498db',    # Blue
    'YOLO': '#e74c3c'        # Red
}

MODEL_NAMES = {
    'MediaPipe full': 'MediaPipe',
    'MoveNet multipose': 'MoveNet',
    'YOLOv8-Pose n': 'YOLO'
}


def load_data():
    """Load analysis results"""
    results_path = Path(__file__).parent / 'results' / 'comprehensive_analysis.json'
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_output_dir():
    """Create figures directory"""
    figures_dir = Path(__file__).parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    return figures_dir


def fig1_model_benchmark(data, output_dir):
    """
    Figure 1: Model Performance Comparison
    Bar chart with mean NMPJPE and standard deviation
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    stats = data['model_benchmark']['descriptive_statistics']

    models = [MODEL_NAMES[s['model']] for s in stats]
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    medians = [s['median'] for s in stats]

    x = np.arange(len(models))
    width = 0.6

    bars = ax.bar(x, means, width,
                  color=[COLORS[m] for m in models],
                  edgecolor='black', linewidth=0.5,
                  alpha=0.85)

    # Add error bars (using std/4 for visibility, noting it in caption)
    ax.errorbar(x, means, yerr=[s/4 for s in stds],
                fmt='none', color='black', capsize=5, capthick=1.5)

    # Add value labels on bars
    for i, (bar, mean, median) in enumerate(zip(bars, means, medians)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'(Md: {median:.1f}%)', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

    ax.set_ylabel('NMPJPE (%)')
    ax.set_xlabel('Model')
    ax.set_title('Model Performance on REHAB24-6 Dataset (Clean Videos, n=121)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 25)

    # Add significance annotations
    ax.annotate('', xy=(0, 21), xytext=(1, 21),
                arrowprops=dict(arrowstyle='-', color='black', lw=1))
    ax.text(0.5, 21.5, '***', ha='center', fontsize=12)

    ax.annotate('', xy=(1, 23), xytext=(2, 23),
                arrowprops=dict(arrowstyle='-', color='black', lw=1))
    ax.text(1.5, 23.5, '***', ha='center', fontsize=12)

    # Add note about error bars
    ax.text(0.02, 0.02, 'Error bars: SD/4 for visibility\n*** p < 0.001',
            transform=ax.transAxes, fontsize=8, va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_model_benchmark.png')
    fig.savefig(output_dir / 'fig1_model_benchmark.pdf')
    plt.close()
    print("[OK] Figure 1: Model Benchmark saved")


def fig2_selection_robustness(data, output_dir):
    """
    Figure 2: Selection Strategy Robustness
    Grouped bar chart comparing Clean vs Coach scenarios
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    robustness = data['selection_robustness']['robustness_comparison']

    models = [MODEL_NAMES[r['model']] for r in robustness]
    clean_means = [r['clean_mean'] for r in robustness]
    coach_means = [r['coach_mean'] for r in robustness]
    increases = [r['relative_increase_pct'] for r in robustness]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, clean_means, width,
                   label='Clean (Single Person)',
                   color=[COLORS[m] for m in models],
                   edgecolor='black', linewidth=0.5,
                   alpha=0.9)

    bars2 = ax.bar(x + width/2, coach_means, width,
                   label='With Coach (Multi-Person)',
                   color=[COLORS[m] for m in models],
                   edgecolor='black', linewidth=0.5,
                   alpha=0.4,
                   hatch='///')

    # Add value labels
    for bar, val in zip(bars1, clean_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    for bar, val, inc in zip(bars2, coach_means, increases):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%\n(+{inc:.0f}%)', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('NMPJPE (%)')
    ax.set_xlabel('Model')
    ax.set_title('Selection Strategy Robustness: Single vs Multi-Person Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 80)
    ax.legend(loc='upper left')

    # Add selection strategy annotations
    strategies = data['selection_robustness']['selection_strategies']
    strategy_text = "Selection Strategies:\n"
    strategy_text += "MediaPipe: Torso-Size (most robust)\n"
    strategy_text += "MoveNet: BBox + Score > 0.1\n"
    strategy_text += "YOLO: BBox + Score > 0.3"

    ax.text(0.98, 0.98, strategy_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Key finding annotation
    ax.annotate('Torso-Selection\n2x more robust',
                xy=(0, 45), xytext=(0.8, 55),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_selection_robustness.png')
    fig.savefig(output_dir / 'fig2_selection_robustness.pdf')
    plt.close()
    print("[OK] Figure 2: Selection Robustness saved")


def fig3_camera_perspective(data, output_dir):
    """
    Figure 3: Camera Perspective Effect
    Grouped bar chart for c17 (frontal) vs c18 (lateral)
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    camera_data = data['camera_perspective']['camera_comparison']

    models = [MODEL_NAMES[c['model']] for c in camera_data]
    c17_means = [c['c17_frontal_mean'] for c in camera_data]
    c18_means = [c['c18_lateral_mean'] for c in camera_data]
    differences = [c['difference'] for c in camera_data]
    cohens_d = [c['cohens_d'] for c in camera_data]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, c17_means, width,
                   label='c17 (Frontal, ~15 deg)',
                   color=[COLORS[m] for m in models],
                   edgecolor='black', linewidth=0.5)

    bars2 = ax.bar(x + width/2, c18_means, width,
                   label='c18 (Lateral, ~75 deg)',
                   color=[COLORS[m] for m in models],
                   edgecolor='black', linewidth=0.5,
                   alpha=0.6)

    # Add value labels with effect size
    for i, (b1, b2, diff, d) in enumerate(zip(bars1, bars2, differences, cohens_d)):
        ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.5,
                f'{c17_means[i]:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.5,
                f'{c18_means[i]:.1f}%', ha='center', va='bottom', fontsize=9)

        # Add difference annotation
        mid_x = (b1.get_x() + b2.get_x() + b2.get_width()) / 2
        max_y = max(c17_means[i], c18_means[i])
        ax.text(mid_x, max_y + 3, f'Delta: {diff:+.1f}%\nd={d:.2f}',
                ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_ylabel('NMPJPE (%)')
    ax.set_xlabel('Model')
    ax.set_title('Camera Perspective Effect: Frontal vs Lateral View')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 30)
    ax.legend(loc='upper right')

    # Interpretation note
    note = "Effect Size Interpretation:\n|d| < 0.2: negligible\n|d| < 0.5: small\n|d| < 0.8: medium"
    ax.text(0.02, 0.98, note, transform=ax.transAxes,
            fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_camera_perspective.png')
    fig.savefig(output_dir / 'fig3_camera_perspective.pdf')
    plt.close()
    print("[OK] Figure 3: Camera Perspective saved")


def fig4_per_joint_heatmap(data, output_dir):
    """
    Figure 4: Per-Joint Error Heatmap
    Heatmap showing NMPJPE by joint and model
    """
    joint_data = data['per_joint']['joint_camera_effect']

    # Prepare data for heatmap (c17 data)
    joints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
              'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
              'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

    joint_labels = ['L Shoulder', 'R Shoulder', 'L Elbow', 'R Elbow',
                    'L Wrist', 'R Wrist', 'L Hip', 'R Hip',
                    'L Knee', 'R Knee', 'L Ankle', 'R Ankle']

    models = ['MediaPipe', 'MoveNet', 'YOLO']

    # Create matrix for c17
    matrix_c17 = np.zeros((len(joints), len(models)))
    matrix_c18 = np.zeros((len(joints), len(models)))

    for item in joint_data:
        model_name = MODEL_NAMES[item['model']]
        if model_name in models:
            model_idx = models.index(model_name)
            joint = item['joint']
            if joint in joints:
                joint_idx = joints.index(joint)
                matrix_c17[joint_idx, model_idx] = item['c17_mean']
                matrix_c18[joint_idx, model_idx] = item['c18_mean']

    # Create figure with two heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # Heatmap for c17
    im1 = ax1.imshow(matrix_c17, cmap='RdYlGn_r', aspect='auto', vmin=8, vmax=28)
    ax1.set_xticks(np.arange(len(models)))
    ax1.set_yticks(np.arange(len(joints)))
    ax1.set_xticklabels(models)
    ax1.set_yticklabels(joint_labels)
    ax1.set_title('c17 (Frontal View)')

    # Add text annotations
    for i in range(len(joints)):
        for j in range(len(models)):
            text = ax1.text(j, i, f'{matrix_c17[i, j]:.1f}',
                           ha='center', va='center', color='black', fontsize=9)

    # Heatmap for c18
    im2 = ax2.imshow(matrix_c18, cmap='RdYlGn_r', aspect='auto', vmin=8, vmax=28)
    ax2.set_xticks(np.arange(len(models)))
    ax2.set_yticks(np.arange(len(joints)))
    ax2.set_xticklabels(models)
    ax2.set_yticklabels(joint_labels)
    ax2.set_title('c18 (Lateral View)')

    # Add text annotations
    for i in range(len(joints)):
        for j in range(len(models)):
            text = ax2.text(j, i, f'{matrix_c18[i, j]:.1f}',
                           ha='center', va='center', color='black', fontsize=9)

    # Add colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.6)
    cbar.set_label('NMPJPE (%)')

    fig.suptitle('Per-Joint Error Analysis by Camera Perspective', fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_per_joint_heatmap.png')
    fig.savefig(output_dir / 'fig4_per_joint_heatmap.pdf')
    plt.close()
    print("[OK] Figure 4: Per-Joint Heatmap saved")


def fig5_confidence_calibration(data, output_dir):
    """
    Figure 5: Confidence Calibration
    Binned analysis showing mean error vs confidence
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    binned_data = data['confidence_calibration']['binned_analysis']
    calibration = data['confidence_calibration']['calibration']

    # Group by model
    for model_full, model_short in MODEL_NAMES.items():
        model_bins = [b for b in binned_data if b['model'] == model_full]

        # Skip very low count bins
        model_bins = [b for b in model_bins if b['count'] > 100]

        if model_bins:
            # Extract bin centers and means
            x_vals = []
            y_vals = []
            sizes = []

            for b in model_bins:
                bin_str = b['conf_bin']
                # Parse interval string like "(0.5, 0.7]"
                parts = bin_str.replace('(', '').replace(']', '').split(', ')
                center = (float(parts[0]) + float(parts[1])) / 2
                x_vals.append(center)
                y_vals.append(b['mean'])
                sizes.append(np.sqrt(b['count']) / 10)

            ax.plot(x_vals, y_vals, 'o-',
                   color=COLORS[model_short],
                   label=model_short,
                   linewidth=2, markersize=8)

    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Mean NMPJPE (%)')
    ax.set_title('Confidence Calibration: Does Higher Confidence Mean Lower Error?')
    ax.legend(loc='upper right')
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 50)

    # Add correlation annotations
    corr_text = "Correlations (Conf vs Error):\n"
    for c in calibration:
        model_short = MODEL_NAMES[c['model']]
        corr_text += f"{model_short}: r = {c['correlation']:.3f}\n"

    ax.text(0.02, 0.98, corr_text, transform=ax.transAxes,
            fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Add interpretation
    ax.text(0.98, 0.02,
            "Negative correlation expected:\nHigher confidence -> Lower error\nYOLO shows strongest calibration",
            transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_confidence_calibration.png')
    fig.savefig(output_dir / 'fig5_confidence_calibration.pdf')
    plt.close()
    print("[OK] Figure 5: Confidence Calibration saved")


def fig6_summary_radar(data, output_dir):
    """
    Figure 6: Multi-Criteria Radar Chart
    Comparing models across multiple dimensions
    """
    # Metrics to compare (normalized 0-100, higher is better)
    metrics = ['Accuracy', 'Robustness', 'Camera\nInvariance', 'Calibration']

    # Calculate scores (inverted where lower is better)
    benchmark = data['model_benchmark']['descriptive_statistics']
    robustness = data['selection_robustness']['robustness_comparison']
    camera = data['camera_perspective']['camera_comparison']
    calib = data['confidence_calibration']['calibration']

    scores = {}
    for model_full, model_short in MODEL_NAMES.items():
        # Accuracy: invert NMPJPE (lower error = higher score)
        bench = next(b for b in benchmark if b['model'] == model_full)
        accuracy = 100 - bench['mean'] * 3  # Scale to 0-100 range

        # Robustness: invert relative increase (lower = better)
        rob = next(r for r in robustness if r['model'] == model_full)
        robustness_score = 100 - min(rob['relative_increase_pct'] / 5, 100)

        # Camera invariance: invert difference (lower = better)
        cam = next(c for c in camera if c['model'] == model_full)
        cam_score = 100 - cam['difference'] * 10

        # Calibration: use absolute correlation (higher = better)
        cal = next(c for c in calib if c['model'] == model_full)
        cal_score = abs(cal['correlation']) * 100

        scores[model_short] = [accuracy, robustness_score, cam_score, cal_score]

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model, values in scores.items():
        values = values + values[:1]  # Close the polygon
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=COLORS[model])
        ax.fill(angles, values, alpha=0.25, color=COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Multi-Criteria Model Comparison\n(Higher is Better)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig6_summary_radar.png')
    fig.savefig(output_dir / 'fig6_summary_radar.pdf')
    plt.close()
    print("[OK] Figure 6: Summary Radar saved")


def fig7_selection_strategy_diagram(output_dir):
    """
    Figure 7: Selection Strategy Comparison Diagram
    Visual explanation of BBox vs Torso selection
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scenario setup
    # Patient (closer, smaller bbox but larger torso relative to distance)
    # Coach (further, larger bbox overall)

    # Left: BBox Selection (fails)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.set_title('BBox Selection Strategy\n(MoveNet, YOLO)', fontsize=12, fontweight='bold')

    # Patient bbox (smaller)
    patient_bbox1 = mpatches.Rectangle((1, 2), 2.5, 5,
                                        linewidth=2, edgecolor='green',
                                        facecolor='lightgreen', alpha=0.5)
    ax1.add_patch(patient_bbox1)
    ax1.text(2.25, 7.5, 'Patient\n(Target)', ha='center', fontsize=10)
    ax1.text(2.25, 1.5, 'BBox: 12.5', ha='center', fontsize=9, color='green')

    # Coach bbox (larger)
    coach_bbox1 = mpatches.Rectangle((5.5, 1), 3.5, 7,
                                      linewidth=2, edgecolor='red',
                                      facecolor='lightcoral', alpha=0.5)
    ax1.add_patch(coach_bbox1)
    ax1.text(7.25, 8.5, 'Coach\n(Wrong!)', ha='center', fontsize=10, color='red')
    ax1.text(7.25, 0.5, 'BBox: 24.5', ha='center', fontsize=9, color='red', fontweight='bold')

    # Selection indicator
    ax1.annotate('SELECTED\n(Larger BBox)', xy=(7.25, 4.5), xytext=(7.25, -0.8),
                fontsize=11, ha='center', fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax1.axis('off')

    # Right: Torso Selection (works)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.set_title('Torso Selection Strategy\n(MediaPipe)', fontsize=12, fontweight='bold')

    # Patient (closer = larger torso in image)
    patient_bbox2 = mpatches.Rectangle((1, 2), 2.5, 5,
                                        linewidth=2, edgecolor='green',
                                        facecolor='lightgreen', alpha=0.5)
    ax2.add_patch(patient_bbox2)

    # Patient torso (large)
    patient_torso = mpatches.Rectangle((1.3, 3.5), 1.9, 2.5,
                                        linewidth=3, edgecolor='darkgreen',
                                        facecolor='green', alpha=0.7)
    ax2.add_patch(patient_torso)
    ax2.text(2.25, 7.5, 'Patient\n(Target)', ha='center', fontsize=10)
    ax2.text(2.25, 1.5, 'Torso: 4.75', ha='center', fontsize=9, color='green', fontweight='bold')

    # Coach (further = smaller torso in image)
    coach_bbox2 = mpatches.Rectangle((5.5, 1), 3.5, 7,
                                      linewidth=2, edgecolor='gray',
                                      facecolor='lightgray', alpha=0.5)
    ax2.add_patch(coach_bbox2)

    # Coach torso (smaller due to distance)
    coach_torso = mpatches.Rectangle((6.2, 3), 1.6, 2,
                                      linewidth=2, edgecolor='gray',
                                      facecolor='gray', alpha=0.5)
    ax2.add_patch(coach_torso)
    ax2.text(7.25, 8.5, 'Coach', ha='center', fontsize=10, color='gray')
    ax2.text(7.25, 0.5, 'Torso: 3.2', ha='center', fontsize=9, color='gray')

    # Selection indicator
    ax2.annotate('SELECTED\n(Larger Torso)', xy=(2.25, 4.75), xytext=(2.25, -0.8),
                fontsize=11, ha='center', fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax2.axis('off')

    fig.suptitle('Why Torso-Based Selection is More Robust in Multi-Person Scenarios',
                 fontsize=14, fontweight='bold', y=1.02)

    # Add explanation
    fig.text(0.5, -0.05,
             'Key Insight: The person closer to the camera has a proportionally larger TORSO,\n'
             'even if their total BBox is smaller than someone further away.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / 'fig7_selection_strategy.png', bbox_inches='tight')
    fig.savefig(output_dir / 'fig7_selection_strategy.pdf', bbox_inches='tight')
    plt.close()
    print("[OK] Figure 7: Selection Strategy Diagram saved")


def main():
    """Generate all figures"""
    print("=" * 60)
    print("Generating Publication-Quality Visualizations")
    print("=" * 60)

    data = load_data()
    output_dir = create_output_dir()

    print(f"\nOutput directory: {output_dir}")
    print("-" * 60)

    # Generate all figures
    fig1_model_benchmark(data, output_dir)
    fig2_selection_robustness(data, output_dir)
    fig3_camera_perspective(data, output_dir)
    fig4_per_joint_heatmap(data, output_dir)
    fig5_confidence_calibration(data, output_dir)
    fig6_summary_radar(data, output_dir)
    fig7_selection_strategy_diagram(output_dir)

    print("-" * 60)
    print(f"\nAll figures saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")

    print("\n[DONE] Visualization generation complete!")


if __name__ == '__main__':
    main()
