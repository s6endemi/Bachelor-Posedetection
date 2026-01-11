# Continuation Prompt fuer naechste AI Session

## Projekt-Kontext
Bachelorarbeit: Vergleich von Pose Estimation Modellen (MediaPipe, MoveNet, YOLO) bei verschiedenen Koerper-Rotationswinkeln zur Kamera.

## Wichtige Dateien zum Einlesen
```
ZWISCHENSTAND.md                    # Aktueller Status (ZUERST LESEN!)
docs/02_PROBLEMS_AND_SOLUTIONS.md   # Alle gefundenen Probleme & Loesungen
docs/04_RESULTS.md                  # Ergebnisse
data/evaluation_results.csv         # Saubere Evaluations-Daten
run_evaluation.py                   # Reproduzierbares Evaluation-Script
```

## Aktueller Stand (09.01.2026)

### WICHTIG: Alte Zahlen waren FEHLERHAFT

| Metrik | Alte Doku (FALSCH) | Korrekt |
|--------|-------------------|---------|
| YOLO c17 | 54.9% | **24.6%** |
| Videos >30% | 49.2% | **25.4%** |

Ursache: Bug im alten Evaluator (frame_step=3 nicht beruecksichtigt).

### Korrigierte Haupt-Ergebnisse

| Modell | NMPJPE | Bewertung |
|--------|--------|-----------|
| **MoveNet** | **14.9%** | BESTE WAHL |
| MediaPipe | 15.6% | Gut, robust bei Multi-Person |
| YOLO | 19.2% | c17-Problem |

### Selection-Robustheit (c17 vs c18)

| Modell | c17-c18 Differenz | Selection-Strategie |
|--------|-------------------|---------------------|
| MediaPipe | +4.3% | Torso-Groesse (robust) |
| MoveNet | +6.8% | BBox + Score |
| YOLO | +10.7% | BBox (anfaellig) |

### Coach-Problem Videos (5 Stueck)

| Video | Situation | MediaPipe | MoveNet/YOLO |
|-------|-----------|-----------|--------------|
| PM_010-c17 | Coach interagiert | 82% (versagt) | 70% (versagt) |
| PM_119-c17 | Coach im Bild | 24% (OK) | 73-74% (versagt) |
| PM_121-c17 | Coach im Bild | 31% (OK) | 60-75% (versagt) |
| PM_108-c17 | Coach im Bild | 17% (OK) | 46-51% (versagt) |
| PM_011-c17 | Coach im Bild | 38% (grenzwertig) | 60-66% (versagt) |

**Erkenntnis:** Torso-Selection (MediaPipe) ist robuster als BBox-Selection bei Multi-Person.

### Rotations-Effekt (c18, sauber)

| Rotation | MediaPipe | MoveNet | YOLO |
|----------|-----------|---------|------|
| 30-40 (schraeg) | 10.2% | 9.7% | 10.8% |
| 70-90 (seitlich) | 15.5% | 12.3% | 15.8% |
| **Anstieg** | **+52%** | **+27%** | **+46%** |

## TODO fuer naechste Session

### 1. Video-Kategorisierung
- [ ] Clean Videos identifizieren (kein Coach)
- [ ] Coach-Interaction Videos markieren
- [ ] Separate Analysen durchfuehren

### 2. Statistische Analyse
- [ ] ANOVA: Unterschied zwischen Modellen signifikant?
- [ ] Tukey HSD: Paarweise Vergleiche
- [ ] Regression: Fehler als Funktion der Rotation

### 3. Visualisierungen
- [ ] NMPJPE vs Rotation (3 Kurven)
- [ ] Per-Joint Heatmap
- [ ] Boxplots pro Winkel-Bin

### 4. Thesis schreiben

## Strategisches Framing

**Multi-Person Challenges als Forschungsbeitrag:**
> "Bei 8% der c17-Videos trat ein Therapeut ins Bild. Diese Real-World-Situation offenbart kritische Unterschiede in der Robustheit der Person-Selection-Strategien..."

Das ist **Staerke**, nicht Schwaeche der Arbeit!

## Daten-Struktur

```
data/
├── videos/Ex1-Ex6/          # MP4 Videos (126 total)
├── gt_2d/Ex1-Ex6/           # 2D Ground Truth - Suffix: -30fps
├── gt_3d/Ex1-Ex6/           # 3D Ground Truth - Suffix: -30fps
├── predictions/Ex1-Ex6/     # Inference Output (.npz)
└── evaluation_results.csv   # Saubere Evaluation (NEU!)
```

## Empfehlungen fuer Previa Health

| Empfehlung | Details |
|------------|---------|
| **Modell** | MoveNet - beste Balance aus Genauigkeit und Robustheit |
| **Alternative** | MediaPipe - robuster bei Multi-Person |
| **YOLO** | Nur bei garantiert Single-Person |
| **Seitlich** | Vermeiden (27-52% Fehleranstieg) |
| **Multi-Person** | Warnung anzeigen, Neupositionierung empfehlen |
