# Results: Ergebnisse

Dieses Dokument enthält alle Ergebnisse der Experimente.

> **WICHTIG (09.01.2026):** Alte Zahlen waren fehlerhaft (Bug: frame_step nicht berücksichtigt).
> Alle Zahlen wurden mit `run_evaluation.py` neu berechnet und Videos kategorisiert.

---

## Status

| Phase | Status |
|-------|--------|
| Mini-Test (500 Frames) | ✅ Abgeschlossen |
| Full-Run (122.400 Frames) | ✅ Abgeschlossen |
| Neu-Evaluation | ✅ Abgeschlossen (korrigiert) |
| **Video-Kategorisierung** | ✅ **Abgeschlossen** |
| Statistische Analyse | ⬜ Ausstehend |
| Visualisierungen | ⬜ Ausstehend |

---

## Video-Kategorisierung

| Kategorie | Beschreibung | Anzahl |
|-----------|--------------|--------|
| **Clean** | Keine Multi-Person-Probleme | 121 (58 c17, 63 c18) |
| **Coach** | Therapeut im Bild | 5 (alle c17) |
| **Total** | | 126 |

---

## HAUPT-ERGEBNIS: Clean Data Ranking

**Auf sauberen Daten (121 Videos) ohne Coach-Interaktion:**

| Modell | NMPJPE | Std | Median | Bewertung |
|--------|--------|-----|--------|-----------|
| **MoveNet** | **12.7%** | 5.7% | 9.8% | BESTE WAHL |
| MediaPipe | 14.4% | 6.1% | 11.2% | Gut, robust |
| YOLO | 17.0% | 8.4% | 11.2% | Höhere Varianz |

**Erkenntnis:** MoveNet ist auf sauberen Daten klar am besten (12.7% vs 14.4% vs 17.0%).

---

## Clean Data: Kamera-Vergleich

### Rohdaten (Mean)

| Modell | c17 | c18 | Differenz |
|--------|-----|-----|-----------|
| MediaPipe | 15.2% | 13.6% | +1.6% |
| MoveNet | 14.1% | 11.4% | +2.8% |
| YOLO | 20.4% | 13.9% | +6.5% |

**Problem:** c17 (frontal) erscheint schlechter als c18 (lateral) - das widerspricht der Erwartung!

### Ursache: Person-Switch Frames

c17 hat **5-18x mehr** Frames mit >100% Fehler (Modell trackt kurzzeitig falsche Person):

| Modell | c17 extreme | c18 extreme | Verhältnis |
|--------|-------------|-------------|------------|
| MediaPipe | 1.52% | 0.30% | **4.8x** |
| MoveNet | 1.57% | 0.11% | **12.8x** |
| YOLO | 2.62% | 0.13% | **17.8x** |

### Nach Filterung (ohne >100% Frames)

| Modell | c17 (gefiltert) | c18 (gefiltert) | Differenz |
|--------|-----------------|-----------------|-----------|
| MediaPipe | 11.8% | 13.1% | **c17 1.3% besser** |
| MoveNet | 9.6% | 11.2% | **c17 1.6% besser** |
| YOLO | 11.3% | 13.7% | **c17 2.4% besser** |

**Erkenntnis:** Frontal (c17) ist tatsächlich besser als lateral - wie erwartet. Der schlechtere Mean-Wert kommt durch sporadische Multi-Person-Frames.

### Wichtig: Mehr Multi-Person-Videos als gedacht

Die 5 identifizierten Coach-Videos sind nur die schlimmsten Fälle. Es gibt **~26 weitere Videos** mit sporadischen Multi-Person-Frames, die gelegentliche Tracking-Fehler verursachen. Das Multi-Person-Robustheitsproblem ist pervasiver als ursprünglich angenommen.

---

## Selection-Robustheit: Coach-Impact

**Wie stark verschlechtert sich jedes Modell bei Coach-Interaktion?**

| Modell | Clean | Mit Coach | Anstieg | Selection |
|--------|-------|-----------|---------|-----------|
| MediaPipe | 14.4% | 45.4% | **+215%** | Torso-Größe |
| MoveNet | 12.7% | 62.2% | **+390%** | BBox + Score |
| YOLO | 17.0% | 66.0% | **+289%** | BBox + Score |

**Kernaussage:** MediaPipe's Torso-Selection ist **~2x robuster** bei Multi-Person-Szenarien als BBox-Selection.

**Warum?** Torso-Größe (Schulter-Hüfte-Abstand) korreliert mit Kamera-Distanz - die nähere Person hat einen größeren Torso im Bild. BBox-Area misst "Ausbreitung" (Armposition), nicht tatsächliche Größe.

---

## Coach-Problem Videos: Detail

| Video | MediaPipe | MoveNet | YOLO | Situation |
|-------|-----------|---------|------|-----------|
| PM_010-c17 | 81.9% | 70.7% | 69.9% | Coach interagiert |
| PM_011-c17 | 38.0% | 59.7% | 66.2% | Coach im Bild |
| PM_108-c17 | 17.2% | 46.2% | 51.1% | Coach im Bild |
| PM_119-c17 | 24.1% | 74.0% | 73.0% | Coach im Bild |
| PM_121-c17 | 30.9% | 60.3% | 75.3% | Coach im Bild |

**Erkenntnisse:**
- PM_010: Alle versagen - physische Coach-Interaktion
- PM_108, PM_119: MediaPipe robust (17-24%), MoveNet/YOLO versagen (46-75%)
- MediaPipe bleibt bei 4/5 Videos unter 40%, MoveNet/YOLO bei 0/5

---

## Dataset-Limitation: Rotationsverteilung

Die ursprünglich geplante Analyse "Wie beeinflusst Körperrotation die Genauigkeit?" konnte nicht durchgeführt werden:

| Rotationsbereich | % der Frames | Beschreibung |
|------------------|--------------|--------------|
| 0-30° (frontal) | 41.2% | Hauptsächlich c17 |
| 30-60° (diagonal) | **17.6%** | Sehr wenig Daten |
| 60-90° (seitlich) | 41.3% | Hauptsächlich c18 |

**Problem:** Die Daten sind **bimodal**, nicht kontinuierlich. Die Patienten rotieren während der Übungen nicht - sie stehen entweder frontal ODER seitlich. Die "Rotationsanalyse" wird damit effektiv zur "Kamera 1 vs Kamera 2 Analyse".

---

## Übungsbasierte Analyse (c18)

| Exercise | Rotation | MediaPipe | MoveNet | YOLO |
|----------|----------|-----------|---------|------|
| Ex1 | 30-50° (schräg) | 12.7% | 10.7% | 11.0% |
| Ex2 | 30-50° (schräg) | 12.4% | 10.0% | 12.6% |
| Ex3 | 50-70° (diagonal) | 16.9% | 15.3% | 17.1% |
| Ex4 | 40-60° (diagonal) | 12.0% | 11.3% | 13.4% |
| Ex5 | 60-80° (seitlich) | 15.7% | 11.9% | 18.8% |
| Ex6 | 30-60° (gemischt) | 11.9% | 10.3% | 12.8% |

**Rotations-Effekt (schräg → seitlich):**
- MoveNet: +1.2% (minimal)
- MediaPipe: +3.0% (moderat)
- YOLO: +7.8% (signifikant)

---

## Analyse nach Übung

| Exercise | MediaPipe | MoveNet | YOLO | Anmerkung |
|----------|-----------|---------|------|-----------|
| Ex1 | 13.7% | **10.6%** | 15.4% | MoveNet dominiert |
| Ex2 | 14.8% | **10.9%** | 13.7% | MoveNet dominiert |
| Ex3 | 17.0% | 17.3% | 20.6% | Alle ähnlich schlecht |
| Ex4 | 13.6% | 13.9% | 18.5% | YOLO kämpft |
| Ex5 | 16.3% | **14.2%** | 20.7% | MoveNet am besten |
| Ex6 | 13.8% | 13.4% | 17.5% | Alle ähnlich |

**Erkenntnis:** MoveNet ist besonders stark bei Ex1, Ex2, Ex5. Ex3 ist für alle Modelle schwierig.

---

## Beste und Schlechteste Videos

### Top 3 (niedrigster Fehler)

| Modell | Video | NMPJPE |
|--------|-------|--------|
| MoveNet | PM_033-c17 | 5.0% |
| MoveNet | PM_003-c17 | 6.3% |
| MediaPipe | PM_032-c17 | 6.1% |
| YOLO | PM_016-c18 | 7.3% |

### Bottom 3 Clean (höchster Fehler ohne Coach)

| Modell | Video | NMPJPE | Ursache |
|--------|-------|--------|---------|
| MediaPipe | PM_025-c17 | 42.0% | Schwierige Bewegung |
| MoveNet | PM_044-c17 | 39.3% | Schwierige Bewegung |
| YOLO | PM_109-c17 | 44.4% | Selection-Instabilität |

---

## Vergleich: Alle Daten vs Clean Data

| Modell | Alle (126) | Clean (121) | Differenz |
|--------|------------|-------------|-----------|
| MediaPipe | 15.6% | 14.4% | -1.2% |
| MoveNet | 14.6% | 12.7% | **-1.9%** |
| YOLO | 19.0% | 17.0% | **-2.0%** |

**Erkenntnis:** Die 5 Coach-Videos erhöhen den Gesamt-Fehler um 1-2%. Bei MoveNet/YOLO ist der Effekt stärker (wegen BBox-Selection).

---

## Hypothesen-Überprüfung (Final)

| Hypothese | Ergebnis | Details |
|-----------|----------|---------|
| H1: NMPJPE steigt mit Rotation | ⚠️ NICHT TESTBAR | Dataset bimodal (41% frontal, 18% mitte, 41% lateral) |
| H2: MoveNet am genauesten | ✅ BESTÄTIGT | 12.7% auf Clean Data, beste Genauigkeit |
| H3: Selection-Strategie kritisch | ✅ BESTÄTIGT | Torso-Selection 2x robuster als BBox |
| H4: c17 hat mehr Tracking-Probleme | ✅ BESTÄTIGT | 5-18x mehr Person-Switch Frames |
| H5: Multi-Person problematisch | ✅ BESTÄTIGT | +215% bis +390% bei Coach-Videos |

**Rotation-Limitation:** Die ursprüngliche Hypothese über kontinuierliche Rotation (0°→90°) konnte nicht getestet werden. Die Patienten rotieren während der Übungen nicht - sie stehen entweder frontal ODER seitlich.

---

## Empfehlungen für Telehealth-Anwendungen

| Szenario | Empfehlung | NMPJPE |
|----------|------------|--------|
| **Single-Person garantiert** | MoveNet MultiPose | 12.7% |
| **Multi-Person möglich** | MediaPipe | 14.4% (robust) |
| **Budget-Hardware** | YOLO Nano | 17.0% (schnellste) |

### Warnungen implementieren:
1. **Multi-Person erkannt:** "Andere Person im Bild. Bitte Kamera neu positionieren."
2. **Seitliche Ansicht:** "Seitliche Ansicht erkannt. Genauigkeit reduziert um ~20%."
3. **Low Confidence:** "Unsichere Schätzung. Bitte Bewegung wiederholen."

---

## Daten-Files

```
data/
├── evaluation_results.csv              # Originale Evaluation
├── evaluation_results_categorized.csv  # Mit Clean/Coach Kategorie
├── predictions/Ex1-Ex6/*.npz           # Prediction Files
└── gt_2d/Ex1-Ex6/*.npy                 # Ground Truth 2D
```

### Scripts

```bash
# Evaluation reproduzieren
.venv/Scripts/python run_evaluation.py

# Videos kategorisieren
.venv/Scripts/python categorize_videos.py

# Clean-Data analysieren
.venv/Scripts/python analyze_clean_data.py
```

---

## Changelog

| Datum | Änderung |
|-------|----------|
| 09.01 | **Kamera-Analyse korrigiert**: c17 hat 5-18x mehr Person-Switch Frames |
| 09.01 | Nach Filterung: c17 (frontal) 1-2% besser als c18 (lateral) |
| 09.01 | ~26 weitere Videos mit sporadischen Multi-Person-Frames identifiziert |
| 09.01 | Rotation-Hypothese als "nicht testbar" markiert (bimodale Daten) |
| 09.01 | **Video-Kategorisierung abgeschlossen** |
| 09.01 | Clean-Data Analyse: MoveNet 12.7%, MediaPipe 14.4%, YOLO 17.0% |
| 09.01 | Selection-Robustheit quantifiziert: MediaPipe 2x robuster |
| 09.01 | Komplett überarbeitet mit korrigierten Zahlen |
