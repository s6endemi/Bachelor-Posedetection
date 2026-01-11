"""
Tiefgehende statistische Analyse für Bachelorarbeit:
"Vergleich von Pose Estimation Modellen bei verschiedenen Körper-Rotationswinkeln"

Dieses Script führt alle Analysen durch und speichert Ergebnisse reproduzierbar.

Forschungsfragen:
    RQ1: Wie quantifiziert sich der Einfluss des Rotationswinkels auf die Schätzgenauigkeit?
    RQ2: Unterscheiden sich die Modelle signifikant in ihrer Rotations-Robustheit?
    RQ3: Welche anatomischen Regionen sind am anfälligsten für rotationsbedingte Fehler?
    RQ4: Wie robust sind verschiedene Person-Selection-Strategien bei Multi-Person-Szenarien?

Hypothesen:
    H1: NMPJPE steigt signifikant mit zunehmendem Rotationswinkel (p < 0.05)
    H2: Die Modelle unterscheiden sich signifikant in ihrer Rotations-Robustheit (ANOVA p < 0.05)
    H3: Distale Joints (Handgelenke) zeigen höhere Fehler als proximale (Schultern) bei Rotation
    H4: Torso-basierte Selection ist robuster als BBox-basierte bei Multi-Person

Autor: [Dein Name]
Datum: 09.01.2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Projekt-Imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pose_evaluation.data.keypoint_mapping import (
    COCO_TO_GT_MAPPING,
    COMPARABLE_COCO_INDICES,
    get_comparable_keypoint_names
)


# =============================================================================
# KONFIGURATION
# =============================================================================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Joint-Gruppen für anatomische Analyse
JOINT_GROUPS = {
    'proximal_upper': [5, 6],      # Schultern (COCO indices)
    'distal_upper': [9, 10],       # Handgelenke
    'mid_upper': [7, 8],           # Ellbogen
    'proximal_lower': [11, 12],    # Hüften
    'distal_lower': [15, 16],      # Knöchel
    'mid_lower': [13, 14],         # Knie
}

JOINT_NAMES = {
    5: 'left_shoulder', 6: 'right_shoulder',
    7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist',
    11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee',
    15: 'left_ankle', 16: 'right_ankle',
}

# Rotations-Bins für Analyse
ROTATION_BINS = [0, 15, 30, 45, 60, 75, 90]
ROTATION_LABELS = ['0-15', '15-30', '30-45', '45-60', '60-75', '75-90']

# Coach-Videos (manuell verifiziert)
COACH_VIDEOS = {'PM_010', 'PM_011', 'PM_108', 'PM_119', 'PM_121'}


# =============================================================================
# DATENSTRUKTUREN
# =============================================================================

@dataclass
class FrameResult:
    """Ergebnis für einen einzelnen Frame."""
    video_id: str
    frame_idx: int
    model: str
    rotation_angle: float
    nmpjpe: float
    per_joint_errors: Dict[str, float]
    confidence_mean: float
    is_coach_video: bool


@dataclass
class StatisticalTest:
    """Ergebnis eines statistischen Tests."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    interpretation: str
    significant: bool


# =============================================================================
# DATEN LADEN
# =============================================================================

def load_all_frame_data(
    predictions_dir: Path,
    gt_2d_dir: Path,
    frame_step: int = 3
) -> List[FrameResult]:
    """
    Lädt alle Frame-Level Daten für tiefgehende Analyse.

    Returns:
        Liste von FrameResult für jeden Frame, jedes Modell
    """
    results = []
    npz_files = sorted(predictions_dir.rglob("*.npz"))

    print(f"Lade {len(npz_files)} Prediction-Dateien...")

    for npz_path in npz_files:
        exercise = npz_path.parent.name
        filename = npz_path.stem
        parts = filename.split("-")
        subject_id = parts[0]
        camera = parts[1]
        video_id = f"{subject_id}-{camera}"

        is_coach = subject_id in COACH_VIDEOS and camera == 'c17'

        # Daten laden
        data = np.load(npz_path)

        # GT laden
        gt_path = gt_2d_dir / exercise / f"{subject_id}-{camera}-30fps.npy"
        if not gt_path.exists():
            continue
        gt_2d = np.load(gt_path)

        # Rotation angles
        if 'rotation_angles' not in data.files:
            continue
        rotation_angles = data['rotation_angles']

        # Für jedes Modell
        for model_key in [k for k in data.files if k.startswith('pred_')]:
            predictions = data[model_key]
            model_name = model_key.replace('pred_', '').replace('_', ' ')

            for i in range(len(predictions)):
                gt_idx = i * frame_step
                if gt_idx >= len(gt_2d) or i >= len(rotation_angles):
                    break

                pred = predictions[i]
                gt = gt_2d[gt_idx]
                angle = rotation_angles[i]

                # Torso-Länge berechnen
                shoulder_mid = (gt[7] + gt[12]) / 2
                hip_mid = (gt[16] + gt[21]) / 2
                torso_length = np.linalg.norm(shoulder_mid - hip_mid)

                if torso_length < 10:
                    continue

                # Per-Joint Fehler berechnen
                per_joint_errors = {}
                valid_errors = []
                confidences = []

                for coco_idx in COMPARABLE_COCO_INDICES:
                    gt_idx_joint = COCO_TO_GT_MAPPING[coco_idx]
                    pred_pt = pred[coco_idx, :2]
                    gt_pt = gt[gt_idx_joint]
                    conf = pred[coco_idx, 2]

                    error = np.linalg.norm(pred_pt - gt_pt)
                    normalized_error = error / torso_length * 100

                    joint_name = JOINT_NAMES.get(coco_idx, f'joint_{coco_idx}')
                    per_joint_errors[joint_name] = normalized_error

                    if conf >= 0.5:
                        valid_errors.append(normalized_error)
                    confidences.append(conf)

                if len(valid_errors) == 0:
                    continue

                results.append(FrameResult(
                    video_id=video_id,
                    frame_idx=i,
                    model=model_name,
                    rotation_angle=float(angle),
                    nmpjpe=float(np.mean(valid_errors)),
                    per_joint_errors=per_joint_errors,
                    confidence_mean=float(np.mean(confidences)),
                    is_coach_video=is_coach
                ))

    print(f"Geladen: {len(results)} Frame-Ergebnisse")
    return results


# =============================================================================
# STATISTISCHE TESTS
# =============================================================================

def test_rotation_effect(df: pd.DataFrame) -> Dict[str, StatisticalTest]:
    """
    H1: NMPJPE steigt signifikant mit Rotationswinkel.

    Tests:
        - Pearson Korrelation (linearer Zusammenhang)
        - Spearman Korrelation (monotoner Zusammenhang)
        - Lineare Regression (Quantifizierung)
    """
    results = {}

    for model in df['model'].unique():
        model_df = df[df['model'] == model]

        # Korrelation
        pearson_r, pearson_p = stats.pearsonr(
            model_df['rotation_angle'],
            model_df['nmpjpe']
        )
        spearman_r, spearman_p = stats.spearmanr(
            model_df['rotation_angle'],
            model_df['nmpjpe']
        )

        # Lineare Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            model_df['rotation_angle'],
            model_df['nmpjpe']
        )

        # Interpretation
        effect = "stark" if abs(pearson_r) > 0.5 else "moderat" if abs(pearson_r) > 0.3 else "schwach"

        results[f'{model}_correlation'] = StatisticalTest(
            test_name=f"Pearson Correlation ({model})",
            statistic=pearson_r,
            p_value=pearson_p,
            effect_size=pearson_r**2,  # R²
            interpretation=f"{effect}er positiver Zusammenhang (r={pearson_r:.3f})",
            significant=pearson_p < 0.05
        )

        results[f'{model}_regression'] = StatisticalTest(
            test_name=f"Linear Regression ({model})",
            statistic=slope,
            p_value=p_value,
            effect_size=r_value**2,
            interpretation=f"Pro 10° Rotation: +{slope*10:.2f}% NMPJPE",
            significant=p_value < 0.05
        )

    return results


def test_model_differences(df: pd.DataFrame) -> Dict[str, StatisticalTest]:
    """
    H2: Die Modelle unterscheiden sich signifikant.

    Tests:
        - One-way ANOVA (Gesamtunterschied)
        - Tukey HSD (Paarweise Vergleiche)
        - Cohen's d (Effektstärken)
    """
    results = {}

    # ANOVA
    groups = [df[df['model'] == m]['nmpjpe'].values for m in df['model'].unique()]
    f_stat, p_value = stats.f_oneway(*groups)

    # Effektstärke (Eta-squared)
    ss_between = sum(len(g) * (np.mean(g) - df['nmpjpe'].mean())**2 for g in groups)
    ss_total = sum((df['nmpjpe'] - df['nmpjpe'].mean())**2)
    eta_squared = ss_between / ss_total

    results['anova'] = StatisticalTest(
        test_name="One-way ANOVA (Modelle)",
        statistic=f_stat,
        p_value=p_value,
        effect_size=eta_squared,
        interpretation=f"eta-squared = {eta_squared:.3f} ({'groß' if eta_squared > 0.14 else 'mittel' if eta_squared > 0.06 else 'klein'}er Effekt)",
        significant=p_value < 0.05
    )

    # Paarweise Vergleiche (t-tests mit Bonferroni-Korrektur)
    models = df['model'].unique()
    n_comparisons = len(models) * (len(models) - 1) // 2
    alpha_corrected = 0.05 / n_comparisons

    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            g1 = df[df['model'] == m1]['nmpjpe'].values
            g2 = df[df['model'] == m2]['nmpjpe'].values

            t_stat, p_val = stats.ttest_ind(g1, g2)

            # Cohen's d
            pooled_std = np.sqrt((np.var(g1) + np.var(g2)) / 2)
            cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std

            results[f'ttest_{m1}_vs_{m2}'] = StatisticalTest(
                test_name=f"t-test: {m1} vs {m2}",
                statistic=t_stat,
                p_value=p_val,
                effect_size=cohens_d,
                interpretation=f"Delta = {np.mean(g1) - np.mean(g2):.2f}%, d = {cohens_d:.2f}",
                significant=p_val < alpha_corrected
            )

    return results


def test_joint_vulnerability(df: pd.DataFrame) -> Dict[str, StatisticalTest]:
    """
    H3: Distale Joints zeigen höhere Fehler bei Rotation.

    Vergleicht Fehleranstieg zwischen proximal und distal.
    """
    results = {}

    # Expandiere per_joint_errors
    joint_data = []
    for _, row in df.iterrows():
        for joint, error in row['per_joint_errors'].items():
            joint_data.append({
                'model': row['model'],
                'rotation_angle': row['rotation_angle'],
                'joint': joint,
                'error': error,
                'is_distal': joint in ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle'],
                'is_upper': 'shoulder' in joint or 'elbow' in joint or 'wrist' in joint
            })

    joint_df = pd.DataFrame(joint_data)

    # Vergleich distal vs proximal bei hoher Rotation (>60°)
    high_rot = joint_df[joint_df['rotation_angle'] > 60]

    distal = high_rot[high_rot['is_distal']]['error'].values
    proximal = high_rot[~high_rot['is_distal']]['error'].values

    t_stat, p_val = stats.ttest_ind(distal, proximal)
    cohens_d = (np.mean(distal) - np.mean(proximal)) / np.sqrt((np.var(distal) + np.var(proximal)) / 2)

    results['distal_vs_proximal'] = StatisticalTest(
        test_name="t-test: Distale vs Proximale Joints (>60° Rotation)",
        statistic=t_stat,
        p_value=p_val,
        effect_size=cohens_d,
        interpretation=f"Distal: {np.mean(distal):.1f}%, Proximal: {np.mean(proximal):.1f}%, Delta = {np.mean(distal) - np.mean(proximal):.1f}%",
        significant=p_val < 0.05
    )

    return results


def test_selection_robustness(df: pd.DataFrame) -> Dict[str, StatisticalTest]:
    """
    H4: Torso-Selection ist robuster bei Multi-Person.

    Vergleicht Fehleranstieg bei Coach-Videos zwischen Modellen.
    """
    results = {}

    for model in df['model'].unique():
        model_df = df[df['model'] == model]

        clean = model_df[~model_df['is_coach_video']]['nmpjpe'].values
        coach = model_df[model_df['is_coach_video']]['nmpjpe'].values

        if len(coach) == 0:
            continue

        t_stat, p_val = stats.ttest_ind(clean, coach)

        results[f'{model}_coach_effect'] = StatisticalTest(
            test_name=f"Coach-Effect ({model})",
            statistic=t_stat,
            p_value=p_val,
            effect_size=(np.mean(coach) - np.mean(clean)) / np.mean(clean) * 100,
            interpretation=f"Clean: {np.mean(clean):.1f}%, Coach: {np.mean(coach):.1f}%, Anstieg: +{(np.mean(coach)/np.mean(clean)-1)*100:.1f}%",
            significant=p_val < 0.05
        )

    return results


# =============================================================================
# ROTATION-ANALYSE
# =============================================================================

def analyze_rotation_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Analysiert NMPJPE nach Rotations-Bins."""

    df = df.copy()
    df['rotation_bin'] = pd.cut(
        df['rotation_angle'],
        bins=ROTATION_BINS,
        labels=ROTATION_LABELS,
        include_lowest=True
    )

    # Aggregieren
    summary = df.groupby(['model', 'rotation_bin']).agg({
        'nmpjpe': ['mean', 'std', 'count'],
        'confidence_mean': 'mean'
    }).round(2)

    summary.columns = ['mean_nmpjpe', 'std_nmpjpe', 'n_frames', 'mean_confidence']

    return summary.reset_index()


def fit_rotation_model(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Fittet Regressionsmodelle für Error vs Rotation.

    Testet:
        - Linear: Error = β₀ + β₁*Rotation
        - Quadratisch: Error = β₀ + β₁*Rotation + β₂*Rotation²
    """
    results = {}

    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        x = model_df['rotation_angle'].values
        y = model_df['nmpjpe'].values

        # Linear
        slope, intercept, r_linear, p_linear, _ = stats.linregress(x, y)

        # Quadratisch
        coeffs = np.polyfit(x, y, 2)
        y_pred_quad = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred_quad)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_quad = 1 - ss_res / ss_tot

        results[model] = {
            'linear': {
                'intercept': intercept,
                'slope': slope,
                'r_squared': r_linear**2,
                'interpretation': f"NMPJPE = {intercept:.2f} + {slope:.3f} × Rotation"
            },
            'quadratic': {
                'coefficients': coeffs.tolist(),
                'r_squared': r_quad,
                'interpretation': f"NMPJPE = {coeffs[2]:.2f} + {coeffs[1]:.3f}×R + {coeffs[0]:.5f}×R²"
            },
            'better_fit': 'quadratic' if r_quad > r_linear**2 + 0.01 else 'linear'
        }

    return results


# =============================================================================
# PER-JOINT ANALYSE
# =============================================================================

def analyze_per_joint(df: pd.DataFrame) -> pd.DataFrame:
    """Detaillierte Per-Joint Analyse."""

    joint_data = []
    for _, row in df.iterrows():
        for joint, error in row['per_joint_errors'].items():
            joint_data.append({
                'model': row['model'],
                'rotation_angle': row['rotation_angle'],
                'joint': joint,
                'error': error
            })

    joint_df = pd.DataFrame(joint_data)

    # Nach Rotation-Bins gruppieren
    joint_df['rotation_bin'] = pd.cut(
        joint_df['rotation_angle'],
        bins=[0, 30, 60, 90],
        labels=['0-30', '30-60', '60-90']
    )

    # Pivot-Tabelle
    summary = joint_df.groupby(['model', 'joint', 'rotation_bin'])['error'].agg(['mean', 'std']).round(2)

    return summary.reset_index()


# =============================================================================
# HAUPTANALYSE
# =============================================================================

def run_full_analysis(
    predictions_dir: Path,
    gt_2d_dir: Path,
    output_dir: Path = RESULTS_DIR
):
    """
    Führt die vollständige Analyse durch und speichert alle Ergebnisse.
    """
    print("=" * 70)
    print("TIEFGEHENDE STATISTISCHE ANALYSE")
    print("Bachelorarbeit: Pose Estimation bei Körper-Rotation")
    print("=" * 70)

    # 1. Daten laden
    print("\n[1/6] Lade Frame-Level Daten...")
    frame_results = load_all_frame_data(predictions_dir, gt_2d_dir)

    # In DataFrame konvertieren
    df = pd.DataFrame([{
        'video_id': r.video_id,
        'frame_idx': r.frame_idx,
        'model': r.model,
        'rotation_angle': r.rotation_angle,
        'nmpjpe': r.nmpjpe,
        'per_joint_errors': r.per_joint_errors,
        'confidence_mean': r.confidence_mean,
        'is_coach_video': r.is_coach_video
    } for r in frame_results])

    # Clean-Daten (ohne Coach)
    df_clean = df[~df['is_coach_video']]

    print(f"   Total Frames: {len(df):,}")
    print(f"   Clean Frames: {len(df_clean):,}")
    print(f"   Coach Frames: {len(df) - len(df_clean):,}")

    # 2. Statistische Tests
    print("\n[2/6] Führe statistische Tests durch...")

    all_tests = {}

    # H1: Rotation Effect
    print("   - H1: Rotationseffekt...")
    all_tests['rotation_effect'] = test_rotation_effect(df_clean)

    # H2: Model Differences
    print("   - H2: Modell-Unterschiede...")
    all_tests['model_differences'] = test_model_differences(df_clean)

    # H3: Joint Vulnerability
    print("   - H3: Joint-Anfälligkeit...")
    all_tests['joint_vulnerability'] = test_joint_vulnerability(df_clean)

    # H4: Selection Robustness
    print("   - H4: Selection-Robustheit...")
    all_tests['selection_robustness'] = test_selection_robustness(df)

    # 3. Rotation-Analyse
    print("\n[3/6] Analysiere Rotations-Bins...")
    rotation_summary = analyze_rotation_bins(df_clean)

    # 4. Regressionsmodelle
    print("\n[4/6] Fitte Regressionsmodelle...")
    regression_models = fit_rotation_model(df_clean)

    # 5. Per-Joint Analyse
    print("\n[5/6] Analysiere Per-Joint Fehler...")
    joint_summary = analyze_per_joint(df_clean)

    # 6. Ergebnisse speichern
    print("\n[6/6] Speichere Ergebnisse...")

    # DataFrame speichern (per_joint_errors als JSON string)
    df_save = df.copy()
    df_save['per_joint_errors'] = df_save['per_joint_errors'].apply(json.dumps)
    df_save.to_csv(output_dir / "frame_level_data.csv", index=False)
    rotation_summary.to_csv(output_dir / "rotation_summary.csv", index=False)
    joint_summary.to_csv(output_dir / "joint_summary.csv", index=False)

    # Statistische Tests als JSON (numpy types konvertieren)
    def convert_to_serializable(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(i) for i in obj]
        return obj

    tests_json = {}
    for category, tests_dict in all_tests.items():
        tests_json[category] = {
            name: convert_to_serializable(asdict(test)) for name, test in tests_dict.items()
        }

    with open(output_dir / "statistical_tests.json", 'w') as f:
        json.dump(tests_json, f, indent=2)

    # Regression Models
    with open(output_dir / "regression_models.json", 'w') as f:
        json.dump(regression_models, f, indent=2)

    # Zusammenfassung ausgeben
    print_summary(df_clean, all_tests, regression_models)

    print(f"\n[OK] Ergebnisse gespeichert in: {output_dir}")

    return df, all_tests, regression_models


def print_summary(df: pd.DataFrame, tests: Dict, regression: Dict):
    """Gibt eine formatierte Zusammenfassung aus."""

    print("\n" + "=" * 70)
    print("ERGEBNIS-ZUSAMMENFASSUNG")
    print("=" * 70)

    # Deskriptive Statistik
    print("\n### Deskriptive Statistik (Clean Data) ###")
    print("\n| Modell | Mean NMPJPE | Std | Median | N Frames |")
    print("|--------|-------------|-----|--------|----------|")
    for model in df['model'].unique():
        m = df[df['model'] == model]['nmpjpe']
        print(f"| {model:20s} | {m.mean():6.2f}% | {m.std():5.2f}% | {m.median():6.2f}% | {len(m):,} |")

    # H1: Rotation Effect
    print("\n### H1: Rotationseffekt ###")
    for name, test in tests['rotation_effect'].items():
        if 'regression' in name:
            print(f"   {test.test_name}")
            print(f"   -> {test.interpretation}")
            print(f"   -> R² = {test.effect_size:.3f}, p < {'0.001' if test.p_value < 0.001 else f'{test.p_value:.3f}'}")

    # H2: Model Differences
    print("\n### H2: Modell-Unterschiede ###")
    anova = tests['model_differences']['anova']
    print(f"   ANOVA: F = {anova.statistic:.2f}, p < {'0.001' if anova.p_value < 0.001 else f'{anova.p_value:.3f}'}")
    print(f"   -> {anova.interpretation}")
    print(f"   -> Signifikant: {'JA' if anova.significant else 'NEIN'}")

    print("\n   Paarweise Vergleiche (Bonferroni-korrigiert):")
    for name, test in tests['model_differences'].items():
        if 'ttest' in name:
            sig = "***" if test.p_value < 0.001 else "**" if test.p_value < 0.01 else "*" if test.significant else ""
            print(f"   -> {test.test_name}: {test.interpretation} {sig}")

    # H3: Joint Vulnerability
    print("\n### H3: Joint-Anfälligkeit ###")
    jv = tests['joint_vulnerability']['distal_vs_proximal']
    print(f"   {jv.interpretation}")
    print(f"   -> Cohen's d = {jv.effect_size:.2f}, p < {'0.001' if jv.p_value < 0.001 else f'{jv.p_value:.3f}'}")
    print(f"   -> Signifikant: {'JA' if jv.significant else 'NEIN'}")

    # H4: Selection Robustness
    print("\n### H4: Selection-Robustheit ###")
    for name, test in tests['selection_robustness'].items():
        print(f"   {test.test_name}")
        print(f"   -> {test.interpretation}")

    # Regression
    print("\n### Regressionsmodelle ###")
    for model, reg in regression.items():
        print(f"\n   {model}:")
        print(f"   -> {reg['linear']['interpretation']}")
        print(f"   -> R² (linear) = {reg['linear']['r_squared']:.3f}")
        print(f"   -> R² (quadr.) = {reg['quadratic']['r_squared']:.3f}")
        print(f"   -> Besserer Fit: {reg['better_fit']}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    predictions_dir = Path("data/predictions")
    gt_2d_dir = Path("data/gt_2d")

    df, tests, regression = run_full_analysis(predictions_dir, gt_2d_dir)
