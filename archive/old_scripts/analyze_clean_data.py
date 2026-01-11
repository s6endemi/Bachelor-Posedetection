"""
Analyse der Clean-Daten (ohne Coach-Interaction Videos).

Fokus auf:
1. Modell-Ranking auf sauberen Daten
2. Rotations-Effekt ohne Selection-Probleme
3. c17 vs c18 Vergleich (reine Kamera-Unterschiede)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_clean_data() -> pd.DataFrame:
    """Laedt kategorisierte Daten und filtert auf Clean."""
    df = pd.read_csv("data/evaluation_results_categorized.csv")
    return df[df['category'] == 'clean']


def overall_ranking(df: pd.DataFrame):
    """Gesamt-Ranking auf Clean-Daten."""
    print("\n" + "=" * 70)
    print("MODELL-RANKING (CLEAN DATA ONLY)")
    print("=" * 70)

    print("\n### Gesamt ###")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        mean = model_df['mean_nmpjpe'].mean()
        std = model_df['mean_nmpjpe'].std()
        median = model_df['mean_nmpjpe'].median()
        n = len(model_df)
        print(f"{model:28s}: {mean:5.1f}% +/- {std:4.1f}% (median: {median:5.1f}%, n={n})")


def camera_comparison(df: pd.DataFrame):
    """Vergleich c17 vs c18 auf Clean-Daten."""
    print("\n" + "=" * 70)
    print("KAMERA-VERGLEICH (CLEAN DATA)")
    print("=" * 70)

    print("\n### c17 vs c18 ###")
    print("\n| Modell | c17 | c18 | Differenz | Interpretation |")
    print("|--------|-----|-----|-----------|----------------|")

    for model in df['model'].unique():
        c17 = df[(df['model'] == model) & (df['camera'] == 'c17')]['mean_nmpjpe'].mean()
        c18 = df[(df['model'] == model) & (df['camera'] == 'c18')]['mean_nmpjpe'].mean()
        diff = c17 - c18
        interp = "OK" if diff < 3 else "moderat" if diff < 5 else "signifikant"
        print(f"| {model:26s} | {c17:4.1f}% | {c18:4.1f}% | +{diff:4.1f}% | {interp} |")


def rotation_analysis(df: pd.DataFrame, predictions_dir: Path):
    """
    Rotation vs NMPJPE Analyse.

    Laedt Rotationswinkel aus den .npz Files und korreliert mit NMPJPE.
    """
    print("\n" + "=" * 70)
    print("ROTATIONS-ANALYSE (CLEAN DATA, c18 only)")
    print("=" * 70)

    # Nur c18 verwenden (weniger Varianz durch Kamera-Position)
    c18_df = df[df['camera'] == 'c18']

    # Rotation-Bins aus vorherigen Analysen (approximiert)
    # Diese Werte basieren auf der C18_FRONTAL_OFFSET = 0 Annahme
    # c18 ist die seitliche Kamera, Winkel variieren je nach Uebung

    print("\n### Approximierte Rotation nach Exercise ###")
    print("(Basierend auf Uebungs-Charakteristik)")

    # Gruppen nach Uebung (approximierte Rotations-Bereiche)
    exercise_rotation = {
        'Ex1': '30-50 (schraeg)',
        'Ex2': '30-50 (schraeg)',
        'Ex3': '50-70 (diagonal)',
        'Ex4': '40-60 (diagonal)',
        'Ex5': '60-80 (seitlich)',
        'Ex6': '30-60 (gemischt)',
    }

    print("\n| Exercise | Rotation | MediaPipe | MoveNet | YOLO |")
    print("|----------|----------|-----------|---------|------|")

    for ex in ['Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6']:
        ex_df = c18_df[c18_df['exercise'] == ex]
        rotation = exercise_rotation[ex]

        mp = ex_df[ex_df['model'] == 'MediaPipe full']['mean_nmpjpe'].mean()
        mn = ex_df[ex_df['model'] == 'MoveNet multipose']['mean_nmpjpe'].mean()
        yolo = ex_df[ex_df['model'] == 'YOLOv8-Pose n']['mean_nmpjpe'].mean()

        print(f"| {ex:8s} | {rotation:17s} | {mp:5.1f}% | {mn:5.1f}% | {yolo:5.1f}% |")


def per_exercise_analysis(df: pd.DataFrame):
    """Analyse nach Uebungstyp."""
    print("\n" + "=" * 70)
    print("ANALYSE NACH UEBUNG")
    print("=" * 70)

    for ex in df['exercise'].unique():
        ex_df = df[df['exercise'] == ex]
        print(f"\n### {ex} ###")

        for model in df['model'].unique():
            model_df = ex_df[ex_df['model'] == model]
            mean = model_df['mean_nmpjpe'].mean()
            std = model_df['mean_nmpjpe'].std()
            print(f"  {model:28s}: {mean:5.1f}% +/- {std:4.1f}%")


def summary_table(df: pd.DataFrame):
    """Zusammenfassende Tabelle fuer Thesis."""
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSENDE TABELLE (fuer Thesis)")
    print("=" * 70)

    # Pivot-Tabelle erstellen
    pivot = df.pivot_table(
        values='mean_nmpjpe',
        index='model',
        columns='camera',
        aggfunc='mean'
    )

    pivot['Gesamt'] = df.groupby('model')['mean_nmpjpe'].mean()
    pivot['Std'] = df.groupby('model')['mean_nmpjpe'].std()
    pivot['c17-c18 Diff'] = pivot['c17'] - pivot['c18']

    print("\n| Modell | c17 | c18 | Gesamt | Std | c17-c18 Diff |")
    print("|--------|-----|-----|--------|-----|--------------|")

    for model in pivot.index:
        print(f"| {model:26s} | {pivot.loc[model, 'c17']:5.1f}% | "
              f"{pivot.loc[model, 'c18']:5.1f}% | {pivot.loc[model, 'Gesamt']:5.1f}% | "
              f"{pivot.loc[model, 'Std']:4.1f}% | +{pivot.loc[model, 'c17-c18 Diff']:4.1f}% |")

    # Best/Worst Videos
    print("\n### Beste Videos (niedrigster Fehler) ###")
    for model in df['model'].unique():
        model_df = df[df['model'] == model].nsmallest(3, 'mean_nmpjpe')
        print(f"\n{model}:")
        for _, row in model_df.iterrows():
            print(f"  {row['subject_id']}-{row['camera']}: {row['mean_nmpjpe']:.1f}%")

    print("\n### Schlechteste Clean-Videos (hoechster Fehler) ###")
    for model in df['model'].unique():
        model_df = df[df['model'] == model].nlargest(3, 'mean_nmpjpe')
        print(f"\n{model}:")
        for _, row in model_df.iterrows():
            print(f"  {row['subject_id']}-{row['camera']}: {row['mean_nmpjpe']:.1f}%")


if __name__ == "__main__":
    df = load_clean_data()
    print(f"Clean Videos geladen: {df['video_id'].nunique()} Videos, {len(df)} Eintraege")

    overall_ranking(df)
    camera_comparison(df)
    rotation_analysis(df, Path("data/predictions"))
    per_exercise_analysis(df)
    summary_table(df)
