# Pose Estimation Evaluation - Vollständige Ergebnisübersicht

> **Letzte Aktualisierung:** 11. Januar 2026
> **Status:** Analyse abgeschlossen
> **Datengrundlage:** 363.529 Frames aus 126 Videos

---

## Executive Summary

Diese Evaluation vergleicht drei Pose-Estimation-Modelle (MediaPipe, MoveNet, YOLOv8-Pose) auf dem REHAB24-6 Dataset. Die wichtigsten Erkenntnisse:

| Erkenntnis | Details |
|------------|---------|
| **Beste Genauigkeit** | MoveNet (11.5% NMPJPE bereinigt) |
| **Robusteste Selection** | MediaPipe (2x besser bei Multi-Person) |
| **Rotationseffekt** | +27-38% Fehleranstieg bei seitlicher Ansicht |
| **Hauptproblem** | c17-Kamera hat 10x mehr Person-Switch-Frames |

---

## 1. Datenbasis

### 1.1 Dataset-Übersicht

| Eigenschaft | Wert |
|-------------|------|
| Dataset | REHAB24-6 (Zenodo) |
| Videos | 126 (21 Patienten × 6 Übungen) |
| Kameras | 2 (c17: frontal, c18: lateral) |
| Analysierte Frames | 363.529 |
| Frame-Step | 3 (jeder 3. Frame bei 30fps = 10Hz) |
| Ground Truth | Motion Capture (optische Marker) |

### 1.2 Video-Kategorisierung

| Kategorie | Anzahl | Beschreibung |
|-----------|--------|--------------|
| Clean | 121 | Keine permanente zweite Person |
| Coach | 5 | Therapeut dauerhaft im Bild |

**Wichtig:** Auch "Clean"-Videos haben sporadische Multi-Person-Frames (siehe Abschnitt 4).

### 1.3 Modelle

| Modell | Entwickler | Keypoints | Selection-Strategie |
|--------|------------|-----------|---------------------|
| MediaPipe Pose | Google | 33 → 12 | Torso-Größe |
| MoveNet MultiPose | Google/TF | 17 → 12 | BBox-Fläche |
| YOLOv8-Pose Nano | Ultralytics | 17 → 12 | BBox-Fläche |

**Vergleichbare Keypoints:** 12 (Schultern, Ellbogen, Handgelenke, Hüften, Knie, Knöchel)

---

## 2. Modell-Ranking

### 2.1 Gesamtergebnis (Clean Data, n=351.778 Frames)

| Modell | Mean | Median | Std | IQR | P90 | P95 |
|--------|------|--------|-----|-----|-----|-----|
| MediaPipe | 14.5% | 11.2% | 23.1% | 3.8% | 17.0% | 22.8% |
| **MoveNet** | **14.8%** | **10.4%** | 32.6% | 4.3% | 16.8% | 20.9% |
| YOLO | 17.7% | 11.3% | 42.5% | 4.8% | 20.2% | 27.4% |

**Interpretation:**
- **Median** ist aussagekräftiger als Mean (robust gegen Ausreißer)
- MoveNet hat besten Median (10.4%)
- YOLO's hoher Mean (17.7%) kommt von extremen Ausreißern (Std=42.5%)

### 2.2 Bereinigt (ohne >100% Frames)

Nach Entfernung von Person-Switch-Frames (~1.1% aller Frames):

| Modell | Mean | Median | Std |
|--------|------|--------|-----|
| **MoveNet** | **11.5%** | **10.4%** | **5.5%** |
| MediaPipe | 12.5% | 11.2% | 7.2% |
| YOLO | 12.9% | 11.3% | 6.8% |

**Erkenntnis:** Die "wahre" Genauigkeit der Modelle ist sehr ähnlich (11-13%). Die Unterschiede im Raw-Mean kommen fast ausschließlich von Ausreißern.

### 2.3 Statistische Signifikanz

**ANOVA:** F=316.30, p<0.001 → Signifikanter Unterschied zwischen Modellen

| Vergleich | Mean Diff | Cohen's d | p-Wert | Signifikant |
|-----------|-----------|-----------|--------|-------------|
| MediaPipe vs MoveNet | -0.25% | -0.009 | 0.098 | Nein |
| MediaPipe vs YOLO | -3.15% | -0.092 | <0.001 | Ja |
| MoveNet vs YOLO | -2.90% | -0.077 | <0.001 | Ja |

**Interpretation:**
- MediaPipe und MoveNet sind statistisch nicht unterscheidbar
- Beide sind signifikant besser als YOLO
- Effect Sizes sind klein (Cohen's d < 0.1)

---

## 3. Kamera-Vergleich (c17 vs c18)

### 3.1 Grundlegende Statistiken

| Modell | c17 Mean | c17 Median | c18 Mean | c18 Median | Diff (Mean) |
|--------|----------|------------|----------|------------|-------------|
| MediaPipe | 15.5% | 10.8% | 13.6% | 11.7% | +1.8% |
| MoveNet | 16.8% | 9.7% | 12.8% | 11.2% | +4.0% |
| YOLO | 21.3% | 10.5% | 14.5% | 12.1% | +6.8% |

**Paradox:** c17 (frontal) hat höheren Mean als c18 (lateral), obwohl frontal einfacher sein sollte.

### 3.2 Ursache: Person-Switch-Frames

| Modell | c17 >100% | c17 Rate | c18 >100% | c18 Rate | Ratio |
|--------|-----------|----------|-----------|----------|-------|
| MediaPipe | 906 | 1.59% | 187 | 0.31% | **5.1x** |
| MoveNet | 1,213 | 2.12% | 95 | 0.16% | **13.3x** |
| YOLO | 1,500 | 2.79% | 79 | 0.13% | **21.5x** |

**Erkenntnis:** c17 hat 5-21x mehr Frames mit >100% Fehler. Das sind Person-Switch-Events, bei denen das Modell kurzzeitig eine andere Person trackt.

### 3.3 Bereinigter Vergleich

Nach Entfernung der >100% Frames:

| Modell | c17 Raw | c17 Clean | Diff | c18 Raw | c18 Clean | Diff |
|--------|---------|-----------|------|---------|-----------|------|
| MediaPipe | 15.5% | 12.0% | -3.5% | 13.6% | 13.1% | -0.5% |
| MoveNet | 16.8% | 10.3% | -6.5% | 12.8% | 12.6% | -0.2% |
| YOLO | 21.3% | 11.3% | -9.9% | 14.5% | 14.3% | -0.2% |

**Erkenntnis:** Nach Bereinigung ist c17 (frontal) ~1-2% besser als c18 (lateral) - wie erwartet.

---

## 4. Ausreißer-Analyse

### 4.1 Verteilung der >100% Frames

| Dimension | Wert |
|-----------|------|
| Total >100% Frames | 3,980 (1.13% aller Clean-Frames) |
| Davon c17 | 3,619 (91%) |
| Davon c18 | 361 (9%) |

### 4.2 Nach Modell

| Modell | N >100% | Rate |
|--------|---------|------|
| MediaPipe | 1,093 | 0.92% |
| MoveNet | 1,308 | 1.10% |
| YOLO | 1,579 | 1.37% |

**Erkenntnis:** YOLO hat 50% mehr extreme Ausreißer als MediaPipe.

### 4.3 Problematische Videos (Top 10)

| Video | N >100% Frames | % des Videos |
|-------|----------------|--------------|
| PM_025-c17 | 322 | 12.4% |
| PM_021-c17 | 237 | 8.3% |
| PM_014-c17 | 229 | 4.4% |
| PM_124-c17 | 188 | 5.7% |
| PM_027-c17 | 185 | 4.5% |
| PM_023-c17 | 181 | 11.0% |
| PM_112-c17 | 164 | 4.3% |
| PM_043-c17 | 139 | 4.0% |
| PM_022-c17 | 134 | 4.9% |
| PM_044-c17 | 113 | 7.0% |

**Alle Top-10 sind c17-Videos.** 72 von 121 Clean-Videos haben mindestens einen >100% Frame.

---

## 5. Rotations-Analyse

### 5.1 Datenverteilung

| Rotation | N Frames | % |
|----------|----------|---|
| 0-30° (frontal) | ~47.000 | 40% |
| 30-60° (diagonal) | ~21.000 | 18% |
| 60-90° (seitlich) | ~50.000 | 42% |

**Limitation:** Die Verteilung ist bimodal (frontal ODER seitlich), nicht kontinuierlich. Die Patienten rotieren nicht während der Übungen.

### 5.2 Rotation nach Kamera (Median NMPJPE)

**c17 (überwiegend 0-70°):**

| Bucket | MediaPipe | MoveNet | YOLO | N Frames |
|--------|-----------|---------|------|----------|
| 0-10° | 10.1% | 9.4% | 9.6% | ~15.700 |
| 10-20° | 10.4% | 8.5% | 9.3% | ~13.700 |
| 50-60° | 10.8% | 10.0% | 11.0% | ~8.500 |
| 60-70° | 12.4% | 10.7% | 11.9% | ~14.800 |

**c18 (überwiegend 20-90°):**

| Bucket | MediaPipe | MoveNet | YOLO | N Frames |
|--------|-----------|---------|------|----------|
| 20-30° | 10.5% | 10.0% | 10.8% | ~15.100 |
| 30-40° | 10.6% | 9.9% | 10.7% | ~9.000 |
| 70-80° | 12.4% | 12.2% | 13.1% | ~16.400 |
| 80-90° | 13.3% | 13.8% | 14.9% | ~16.200 |

### 5.3 Rotationseffekt (c18-only, saubere Daten)

| Modell | Schräg (20-40°) | Seitlich (70-90°) | Anstieg |
|--------|-----------------|-------------------|---------|
| MediaPipe | 10.5% | 13.3% | **+27%** |
| MoveNet | 10.0% | 13.8% | **+38%** |
| YOLO | 10.7% | 14.9% | **+39%** |

**Erkenntnis:** Der Rotationseffekt ist real und messbar. Bei seitlicher Ansicht steigt der Fehler um 27-39% relativ.

---

## 6. Selection-Robustheit (Coach-Impact)

### 6.1 Coach vs Clean Vergleich

| Modell | Clean Mean | Coach Mean | Anstieg | Anstieg % |
|--------|------------|------------|---------|-----------|
| **MediaPipe** | 14.5% | 44.8% | +30.3% | **+209%** |
| MoveNet | 14.8% | 64.9% | +50.2% | +340% |
| YOLO | 17.7% | 66.1% | +48.5% | +274% |

### 6.2 Interpretation

MediaPipe's Torso-basierte Selection ist **~2x robuster** als BBox-Selection bei Multi-Person-Szenarien.

**Warum?**
- **Torso-Größe** (Schulter-Hüfte-Abstand) korreliert mit Kamera-Distanz
- **BBox-Fläche** misst "Ausbreitung" (Armposition), nicht tatsächliche Größe
- Ein Coach mit ausgestreckten Armen hat große BBox, aber kleinen Torso im Vergleich

---

## 7. Per-Joint Analyse

### 7.1 Fehler nach Körperregion (Clean Data, Median)

| Region | MediaPipe | MoveNet | YOLO |
|--------|-----------|---------|------|
| Schultern | 7-8% | 8-10% | 8-10% |
| Ellbogen | 9% | 10% | 10% |
| Handgelenke | 9-10% | 11-12% | 12-13% |
| Hüften | 16-17% | 10-11% | 14-15% |
| Knie | 7-8% | 6% | 7-8% |
| Knöchel | 12% | 10% | 10% |

### 7.2 Auffälligkeiten

- **MediaPipe:** Überdurchschnittlich hoher Hüft-Fehler (16-17% vs 10-15% bei anderen)
- **Alle Modelle:** Handgelenke haben höheren Fehler als Schultern
- **Knie:** Niedrigster Fehler bei allen Modellen (6-8%)

---

## 8. Übungs-Analyse

### 8.1 Median NMPJPE nach Übung

| Übung | MediaPipe | MoveNet | YOLO | Schwierigkeit |
|-------|-----------|---------|------|---------------|
| Ex1 | 11.4% | 10.2% | 10.4% | Leicht |
| Ex2 | 10.8% | 9.4% | 10.4% | **Leichteste** |
| Ex3 | 12.6% | 13.6% | 14.2% | **Schwierigste** |
| Ex4 | 11.3% | 11.2% | 12.0% | Mittel |
| Ex5 | 11.0% | 11.2% | 13.1% | Mittel |
| Ex6 | 11.3% | 10.2% | 11.6% | Leicht |

### 8.2 Interpretation

- Ex2 ist für alle Modelle am einfachsten (9-10%)
- Ex3 ist für alle Modelle am schwierigsten (13-14%)
- Die Übungs-Unterschiede sind moderat (±2-3%)

---

## 9. Methodische Erkenntnisse

### 9.1 Selection-Strategien

| Modell | Strategie | Funktioniert bei Multi-Person |
|--------|-----------|-------------------------------|
| MediaPipe | Torso-Größe | **Gut** (2x robuster) |
| MoveNet | BBox-Fläche | Anfällig |
| YOLO | BBox-Fläche | Anfällig |

### 9.2 Confidence-Thresholds

| Modell | Verwendeter Threshold | Begründung |
|--------|----------------------|------------|
| MediaPipe | 0.1 (Detection) | Default 0.5 zu strikt (29% Failures) |
| MoveNet | 0.1 (Score) | Niedrig für Robustheit |
| YOLO | 0.3 (Joint Confidence) | Balance Robustheit/Qualität |

### 9.3 Frame-Alignment

**Kritisch:** Bei `frame_step=3` muss Evaluation den gleichen Step verwenden:
```
Prediction[i] entspricht Video-Frame[i * 3]
```

---

## 10. Limitationen

| Limitation | Auswirkung |
|------------|------------|
| Bimodale Rotationsverteilung | Keine kontinuierliche Rotation-Analyse möglich |
| Nur 5 Coach-Videos | Selection-Robustheit basiert auf kleiner Stichprobe |
| c17 Multi-Person-Problem | Verfälscht Kamera-Vergleich ohne Bereinigung |
| Nur mobile Modellvarianten | Keine Aussage über Server-Varianten |

---

## 11. Empfehlungen für Anwendungen

### 11.1 Modellwahl

| Szenario | Empfehlung | Begründung |
|----------|------------|------------|
| Single-Person garantiert | MoveNet | Beste Genauigkeit (10.4% Median) |
| Multi-Person möglich | MediaPipe | 2x robustere Selection |
| Ressourcen-limitiert | YOLO Nano | Schnellste Inferenz |

### 11.2 Implementierungshinweise

1. **Multi-Person-Warnung:** App sollte warnen wenn >1 Person erkannt
2. **Seitliche Ansicht:** Fehler steigt um ~30%, User zur Neupositionierung auffordern
3. **Confidence-Filter:** Niedrige Joint-Confidence → Frame verwerfen
4. **Extreme-Frame-Detection:** NMPJPE >100% → automatisch als ungültig markieren

---

## 12. Dateien und Reproduzierbarkeit

### 12.1 Analyse-Dateien

```
analysis/
├── comprehensive_analysis.py     # Haupt-Analyse-Script
├── extended_analysis.py          # Erweiterte Analysen
├── ANALYSIS_REPORT.md            # Detaillierter Report
├── EXTENDED_ANALYSIS_REPORT.md   # Erweiterte Analysen
├── EVALUATION_SUMMARY.md         # Diese Datei
└── results/
    ├── frame_level_data.csv      # 363k Frames, alle Metriken
    ├── rotation_analysis.csv     # 10°-Buckets
    ├── outlier_analysis.csv      # Video-Level Ausreißer
    ├── per_joint_analysis.csv    # Joint-Level Fehler
    └── summary_statistics.json   # Maschinenlesbar
```

### 12.2 Reproduktion

```bash
# Analyse ausführen
cd C:/Users/Eren/bachelor
.venv/Scripts/python analysis/comprehensive_analysis.py
.venv/Scripts/python analysis/extended_analysis.py
```

### 12.3 Abhängigkeiten

- Python 3.10+
- pandas, numpy, scipy
- Vorhandene Predictions in `data/predictions/`
- Ground Truth in `data/gt_2d/`

---

## Changelog

| Datum | Änderung |
|-------|----------|
| 11.01.2026 | Vollständige Analyse erstellt |
| 11.01.2026 | Frame-Level Daten extrahiert (363k Frames) |
| 11.01.2026 | Rotation nach Kamera getrennt |
| 11.01.2026 | Person-Switch-Bereinigung implementiert |
| 11.01.2026 | Übungs-Analyse hinzugefügt |
