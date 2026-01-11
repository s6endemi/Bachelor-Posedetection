"""
Video-Kategorisierung: Clean vs Coach-Interaction.

Identifiziert Videos mit Coach/Therapeut im Bild basierend auf
der Differenz zwischen MediaPipe und MoveNet/YOLO Performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Bekannte Coach-Problem Videos (manuell identifiziert)
COACH_VIDEOS = {
    'PM_010-c17',  # Coach interagiert direkt mit Patient
    'PM_011-c17',  # Coach im Hintergrund
    'PM_108-c17',  # Coach im Hintergrund
    'PM_119-c17',  # Coach im Hintergrund
    'PM_121-c17',  # Coach im Hintergrund
}


def categorize_videos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kategorisiert Videos in Clean und Coach-Interaction.

    Args:
        df: DataFrame mit evaluation_results

    Returns:
        DataFrame mit zusaetzlicher 'category' Spalte
    """
    # Video-ID erstellen
    df = df.copy()
    df['video_id'] = df['subject_id'] + '-' + df['camera']

    # Kategorie zuweisen
    df['category'] = df['video_id'].apply(
        lambda x: 'coach' if x in COACH_VIDEOS else 'clean'
    )

    return df


def analyze_by_category(df: pd.DataFrame):
    """Analysiert Ergebnisse getrennt nach Kategorie."""

    print("=" * 70)
    print("VIDEO-KATEGORISIERUNG UND ANALYSE")
    print("=" * 70)

    # Uebersicht
    video_counts = df.groupby(['category', 'camera'])['video_id'].nunique()
    print("\n### Video-Verteilung ###")
    print(video_counts)

    # Gesamt-Statistik pro Kategorie
    print("\n### NMPJPE nach Kategorie (alle Modelle) ###")
    for category in ['clean', 'coach']:
        cat_df = df[df['category'] == category]
        print(f"\n{category.upper()} Videos:")
        for model in df['model'].unique():
            model_df = cat_df[cat_df['model'] == model]
            if len(model_df) == 0:
                continue
            mean = model_df['mean_nmpjpe'].mean()
            std = model_df['mean_nmpjpe'].std()
            n = len(model_df)
            print(f"  {model:28s}: {mean:5.1f}% +/- {std:4.1f}% (n={n})")

    # Clean-only Kamera-Vergleich
    print("\n### Clean Videos: c17 vs c18 ###")
    clean_df = df[df['category'] == 'clean']
    for camera in ['c17', 'c18']:
        print(f"\n{camera}:")
        cam_df = clean_df[clean_df['camera'] == camera]
        for model in df['model'].unique():
            model_df = cam_df[cam_df['model'] == model]
            if len(model_df) == 0:
                continue
            mean = model_df['mean_nmpjpe'].mean()
            print(f"  {model:28s}: {mean:5.1f}%")

    # Coach-Videos Detail
    print("\n### Coach-Interaction Videos Detail ###")
    coach_df = df[df['category'] == 'coach']
    for video_id in COACH_VIDEOS:
        print(f"\n{video_id}:")
        vid_df = coach_df[coach_df['video_id'] == video_id]
        for _, row in vid_df.iterrows():
            print(f"  {row['model']:28s}: {row['mean_nmpjpe']:5.1f}%")

    # Selection-Robustheit bei Coach-Videos
    print("\n### Selection-Robustheit bei Coach-Videos ###")
    coach_summary = coach_df.groupby('model')['mean_nmpjpe'].agg(['mean', 'std'])
    clean_c17 = clean_df[clean_df['camera'] == 'c17'].groupby('model')['mean_nmpjpe'].mean()

    print("\n| Modell | Clean c17 | Coach c17 | Differenz |")
    print("|--------|-----------|-----------|-----------|")
    for model in df['model'].unique():
        clean_val = clean_c17[model] if model in clean_c17.index else 0
        coach_val = coach_summary.loc[model, 'mean'] if model in coach_summary.index else 0
        diff = coach_val - clean_val
        print(f"| {model:26s} | {clean_val:5.1f}% | {coach_val:5.1f}% | +{diff:5.1f}% |")


def save_categorized_results(df: pd.DataFrame, output_path: Path):
    """Speichert kategorisierte Ergebnisse."""
    df.to_csv(output_path, index=False)
    print(f"\nKategorisierte Ergebnisse gespeichert: {output_path}")


if __name__ == "__main__":
    # Daten laden
    results_path = Path("data/evaluation_results.csv")
    df = pd.read_csv(results_path)

    # Kategorisieren
    df = categorize_videos(df)

    # Analysieren
    analyze_by_category(df)

    # Speichern
    output_path = Path("data/evaluation_results_categorized.csv")
    save_categorized_results(df, output_path)

    # Zusammenfassung
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)

    n_clean = df[df['category'] == 'clean']['video_id'].nunique()
    n_coach = df[df['category'] == 'coach']['video_id'].nunique()

    print(f"\nClean Videos: {n_clean}")
    print(f"Coach Videos: {n_coach}")
    print(f"Total: {n_clean + n_coach}")
