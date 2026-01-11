# Experiments: Tests und Validierung

Dieses Dokument enthält alle durchgeführten Experimente, deren Setup und Ergebnisse.

---

## Experiment 1: MediaPipe Confidence Threshold

### Hypothese
Der Default-Wert `min_detection_confidence=0.5` ist zu strikt für Real-World Szenarien.

### Setup
- 100 Frames aus einem Video
- MediaPipe Heavy (complexity=2)
- Verschiedene Confidence-Werte

### Ergebnisse

| Confidence | Detection Failures | NMPJPE |
|------------|-------------------|--------|
| 0.5 (Default) | 29% | 128% |
| 0.3 | ~10% | ~40% |
| 0.1 | <1% | ~10% |

### Fazit
`confidence=0.1` ist optimal für robuste Detection ohne signifikanten Genauigkeitsverlust.

---

## Experiment 2: Person-Selection Strategien

### Hypothese
Die Methode zur Person-Auswahl bei Multi-Person-Szenarien beeinflusst die Genauigkeit stark.

### Setup
- Video c18 mit bekannter Hintergrund-Person
- 100 Frames
- Verschiedene Selection-Methoden pro Modell

### YOLO Selection

| Methode | Falsche Person | NMPJPE |
|---------|----------------|--------|
| Erste Person (Index 0) | Häufig | 37% |
| Höchste Confidence | Manchmal | ~20% |
| **Größte BBox** | **Nie** | **~14%** |

### MediaPipe Selection

| Methode | Falsche Person | NMPJPE |
|---------|----------------|--------|
| BBox (alle 33 KP) | 11% | ~25% |
| BBox (12 Body KP) | 8% | ~22% |
| **Torso-Größe** | **2%** | **~12%** |

### Fazit
- **YOLO/MoveNet:** BBox Area ist beste Strategie
- **MediaPipe:** Torso-Größe ist beste Strategie
- **Grund:** MediaPipe hat keine echten BBoxes

---

## Experiment 3: MoveNet SinglePose vs MultiPose

### Hypothese
SinglePose ist ungeeignet für Szenarien mit Hintergrund-Personen.

### Setup
- 5 Videos mit bekannten Multi-Person-Situationen
- 100 Frames pro Video
- Vergleich SinglePose vs MultiPose

### Ergebnisse

| Variante | Architektur | Wrong Person Rate |
|----------|-------------|-------------------|
| SinglePose Thunder | Nur 1 Person | 2.8% |
| **MultiPose Lightning** | Bis 6 Personen | **0%** |

### Qualitative Beobachtung
SinglePose wählt manchmal die "dominantere" Person im Bild, die nicht unbedingt die größte ist.

### Fazit
MultiPose ist notwendig für robuste Multi-Person-Szenarien. SinglePose sollte nur dokumentiert, nicht verwendet werden.

---

## Experiment 4: Mini-Test (Pipeline-Validierung)

### Ziel
Validierung der gesamten Pipeline vor Full-Run.

### Setup
```python
# 5 Videos × 100 Frames = 500 Frames
python run_inference.py --max-videos 5 --max-frames 100 --yolo-size n
```

### Modelle
- MediaPipe Heavy (confidence=0.1, Torso-Selection)
- MoveNet MultiPose (BBox-Selection)
- YOLOv8-Pose Nano (BBox-Selection)

### Ergebnisse: Fehldetektionen

| Modell | Detection Failures | Wrong Person | Total Errors |
|--------|-------------------|--------------|--------------|
| MediaPipe | 2 (0.4%) | 0 | 2 (0.4%) |
| MoveNet MultiPose | 0 (0.0%) | 0 | 0 (0.0%) |
| YOLO | 0 (0.0%) | 0 | 0 (0.0%) |

### Ergebnisse: NMPJPE (vorläufig)

| Winkel-Bin | MediaPipe | MoveNet MP | YOLO |
|------------|-----------|------------|------|
| 0-10° | 10.4% | ~14% | 10.4% |
| 10-20° | 11.2% | ~14% | 11.0% |
| 20-30° | 11.8% | ~14% | 11.5% |
| 30-40° | 12.5% | ~14% | 12.3% |
| 40-50° | 13.6% | ~14% | 13.8% |
| 50-60° | 19.6% | ~14% | 14.3% |
| 60-70° | - | - | - |
| 70-80° | - | - | - |
| 80-90° | - | - | - |

*Hinweis: Nicht alle Winkel-Bins hatten Daten im Mini-Test*

### Fazit
- Pipeline funktioniert korrekt
- Alle Selection-Strategien validiert
- Bereit für Full-Run

---

## Experiment 5: BBox vs Torso für MediaPipe (Detailanalyse)

### Hypothese
Bounding Box aus Keypoints funktioniert anders als echte Detector-BBox.

### Setup
- Frame mit 2 Personen (Haupt + Hintergrund)
- MediaPipe mit num_poses=5
- Vergleich verschiedener Größen-Metriken

### Messwerte (ein Frame)

| Person | BBox (33 KP) | BBox (12 KP) | Torso-Größe |
|--------|--------------|--------------|-------------|
| Hintergrund | 0.1064 | 0.0642 | 0.0956 |
| **Hauptperson** | 0.0807 | 0.0511 | **0.1775** |

### Analyse
```
BBox (alle Keypoints):
- Hintergrund > Hauptperson (FALSCH!)
- Grund: Gesichts-Keypoints verzerren BBox

BBox (nur Body):
- Immer noch: Hintergrund > Hauptperson (FALSCH!)
- Grund: Arm-Spread, nicht Kamera-Distanz

Torso-Größe:
- Hauptperson > Hintergrund (RICHTIG!)
- Grund: Näher = größerer Torso
```

### Visualisierung
```
Hintergrund-Person:        Hauptperson:
     ●                          ●
   / | \    (Arme weit)       / | \
  /  |  \                    ●  |  ●
 ●   |   ●                      |
     |                          |
     ●                          ●  (näher an Kamera)
    / \                        / \
   /   \                      /   \
  ●     ●                    ●     ●

BBox: GRÖSSER              BBox: kleiner
Torso: kleiner             Torso: GRÖSSER
```

### Fazit
BBox aus Keypoints misst **Spread**, nicht **Size**. Torso-Größe ist das korrekte Maß für Kamera-Distanz.

---

## Experiment 6: Rotationswinkel-Verteilung im Dataset

### Ziel
Verstehen, welche Rotationswinkel im Dataset vorkommen.

### Setup
- Rotationswinkel aus 3D GT berechnet
- Über alle Frames aggregiert (Mini-Test)

### Ergebnisse (vorläufig)

```
0-10°:   ████████████████████  (häufig - frontale Übungen)
10-20°:  ██████████████
20-30°:  ████████████
30-40°:  ██████████
40-50°:  ████████
50-60°:  ██████
60-70°:  ████
70-80°:  ██
80-90°:  █  (selten - fast seitlich)
```

### Implikation
- Mehr Daten für frontale Winkel
- Weniger Daten für extreme Rotation
- Muss bei statistischer Analyse berücksichtigt werden

---

## Experiment 7: Full-Run (ABGESCHLOSSEN)

### Setup
```bash
.venv/Scripts/python run_inference.py --yolo-size n --frame-step 3
```

### Parameter
- **Modelle:** MediaPipe Full (complexity=1), MoveNet MultiPose Lightning, YOLOv8-Pose Nano
- **Videos:** 126 (Ex1-Ex6, c17+c18)
- **Frames:** 122.400 (bei frame_step=3, entspricht ~10Hz)
- **Dauer:** ~4-5 Stunden

### Outputs
```
data/predictions/
├── Ex1/
│   ├── PM_000-c17.npz
│   ├── PM_000-c18.npz
│   └── ... (21 Videos)
├── Ex2/ (21 Videos)
├── Ex3/ (21 Videos)
├── Ex4/ (21 Videos)
├── Ex5/ (21 Videos)
└── Ex6/ (21 Videos)
```

### Haupt-Ergebnisse

#### Modell-Ranking (Gesamt)
| Modell | NMPJPE | Std | Bewertung |
|--------|--------|-----|-----------|
| **MoveNet** | **10.8%** | 11.8% | BESTE WAHL |
| MediaPipe | 14.3% | 21.4% | Gut, aber variabel |
| YOLO | 14.6% | 25.0% | Gut, aber Selection-Problem |

#### Rotations-Effekt (c18-only, saubere Daten)
| Rotation | MediaPipe | MoveNet | YOLO | Anstieg |
|----------|-----------|---------|------|---------|
| 25-40° (schraeg) | 10.5% | 10.4% | 11.2% | - |
| 70-90° (seitlich) | 17.5% | 16.7% | 18.5% | **+60-67%** |

#### Kritische Entdeckung: YOLO Coach-Problem
| Kamera | YOLO NMPJPE | Videos >30% Fehler |
|--------|-------------|--------------------|
| c17 | **54.9%** | **49.2%** |
| c18 | 15.3% | 0.0% |

**Root Cause:** YOLO Selection hat keinen Score-Filter (im Gegensatz zu MoveNet).
Coach tritt bei c17-Videos ins Bild, hat groessere BBox, wird faelschlicherweise gewaehlt.

### Analysierte Effekte
1. ✅ NMPJPE vs Rotation: **+60-67% Anstieg bei seitlicher Ansicht**
2. ✅ Per-Joint Analyse: **Handgelenke +67-93%, Hueften -14-39%**
3. ✅ Ausreisser: **31 YOLO Videos >30%, alle c17**
4. ⏳ Statistische Signifikanz: Ausstehend (ANOVA/Tukey)

---

## Experiment-Log

| Datum | Experiment | Ergebnis | Aktion |
|-------|------------|----------|--------|
| 05.01 | MediaPipe Confidence | 0.5 zu strikt | → 0.1 |
| 05.01 | YOLO Selection | Index 0 falsch | → BBox Area |
| 05.01 | MoveNet SinglePose | Architektur-Limitation | → MultiPose |
| 06.01 | MediaPipe BBox | Macht es schlimmer | → Torso bleibt |
| 06.01 | Mini-Test | Pipeline validiert | → Ready for Full-Run |
| 06.01 | Kamera-Koordinatensystem | MoCap ≠ Kamera | → C17_FRONTAL_OFFSET = 65° |
| 07.01 | Full-Run Start | 126 Videos | frame_step=3 |
| 08.01 | Full-Run DONE | 122.400 Frames | Alle .npz gespeichert |
| 08.01 | YOLO Coach-Problem | 31 Videos >30% bei c17 | Root Cause: kein Score-Filter |
| 08.01 | Rotations-Trend | +60-67% bei seitlich | Auf c18-only bestaetigt |
| 08.01 | Per-Joint Analyse | Handgelenke am schlimmsten | +67-93% bei Rotation |
| 08.01 | Modell-Ranking | MoveNet beste Wahl | 10.8% NMPJPE, robuste Selection |

---

## Reproduzierbarkeit

### Alle Experimente sind reproduzierbar mit:
```bash
# Mini-Test
python run_inference.py --max-videos 5 --max-frames 100 --yolo-size n

# Single Video Debug
python -c "
from src.pose_evaluation.estimators import MediaPipeEstimator
# ... Debug code
"
```

### Seeds und Versionen
- Python: 3.10+
- MediaPipe: 0.10.x
- TensorFlow: 2.13+
- Ultralytics: 8.x
- NumPy: 1.24+
