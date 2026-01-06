# Results: Ergebnisse

Dieses Dokument enthält alle Ergebnisse der Experimente. Es wird nach dem Full-Run vervollständigt.

---

## Status

| Phase | Status |
|-------|--------|
| Mini-Test (500 Frames) | ✅ Abgeschlossen |
| Full-Run (~185k Frames) | ⬜ Ausstehend |
| Statistische Analyse | ⬜ Ausstehend |
| Visualisierungen | ⬜ Ausstehend |

---

## Mini-Test Ergebnisse (Vorläufig)

### Datenbasis
- 5 Videos × 100 Frames = 500 Frames
- Modelle: MediaPipe Heavy, MoveNet MultiPose, YOLO Nano

### Detection Performance

| Modell | Detection Failures | Wrong Person | Valid Frames |
|--------|-------------------|--------------|--------------|
| MediaPipe | 2 (0.4%) | 0 | 498 (99.6%) |
| MoveNet MultiPose | 0 (0.0%) | 0 | 500 (100%) |
| YOLO | 0 (0.0%) | 0 | 500 (100%) |

### NMPJPE nach Rotationswinkel (Vorläufig)

| Winkel | MediaPipe | MoveNet MP | YOLO | N |
|--------|-----------|------------|------|---|
| 0-10° | 10.4% | ~14% | 10.4% | ~100 |
| 10-20° | 11.2% | ~14% | 11.0% | ~80 |
| 20-30° | 11.8% | ~14% | 11.5% | ~70 |
| 30-40° | 12.5% | ~14% | 12.3% | ~60 |
| 40-50° | 13.6% | ~14% | 13.8% | ~50 |
| 50-60° | 19.6% | ~14% | 14.3% | ~40 |

*N = Anzahl Frames in diesem Bin (geschätzt)*

### Vorläufige Beobachtungen

1. **MediaPipe** zeigt Degradation bei höheren Winkeln
2. **MoveNet** ist überraschend stabil (~14% konstant)
3. **YOLO** ähnlich wie MediaPipe, aber weniger Streuung
4. **Alle** haben niedrige Fehlerraten (<1%)

---

## Full-Run Ergebnisse

*Wird nach Abschluss des Full-Runs ausgefüllt*

### Geplante Tabellen

#### Haupt-Ergebnis: NMPJPE vs Rotation

```
| Winkel | MediaPipe      | MoveNet MP     | YOLO           |
|--------|----------------|----------------|----------------|
|        | Mean ± Std     | Mean ± Std     | Mean ± Std     |
| 0-10°  |                |                |                |
| 10-20° |                |                |                |
| 20-30° |                |                |                |
| 30-40° |                |                |                |
| 40-50° |                |                |                |
| 50-60° |                |                |                |
| 60-70° |                |                |                |
| 70-80° |                |                |                |
| 80-90° |                |                |                |
```

#### Per-Joint Fehler

```
| Joint          | MediaPipe | MoveNet | YOLO |
|----------------|-----------|---------|------|
| Left Shoulder  |           |         |      |
| Right Shoulder |           |         |      |
| Left Elbow     |           |         |      |
| Right Elbow    |           |         |      |
| Left Wrist     |           |         |      |
| Right Wrist    |           |         |      |
| Left Hip       |           |         |      |
| Right Hip      |           |         |      |
| Left Knee      |           |         |      |
| Right Knee     |           |         |      |
| Left Ankle     |           |         |      |
| Right Ankle    |           |         |      |
```

#### Ausreißer-Statistik

```
| Modell    | Failures | Wrong Person | High Error | Valid |
|-----------|----------|--------------|------------|-------|
| MediaPipe |          |              |            |       |
| MoveNet   |          |              |            |       |
| YOLO      |          |              |            |       |
```

---

## Geplante Visualisierungen

### 1. Hauptgrafik: NMPJPE vs Rotationswinkel
```
Y-Achse: NMPJPE (%)
X-Achse: Rotationswinkel (°)
3 Linien: MediaPipe, MoveNet, YOLO
Shaded Area: Konfidenzintervall (±1 Std)
```

### 2. Per-Joint Heatmap
```
Y-Achse: Joints (12)
X-Achse: Rotationswinkel-Bins (9)
Farbe: NMPJPE (niedrig=grün, hoch=rot)
Eine Heatmap pro Modell
```

### 3. Boxplots pro Winkel-Bin
```
X-Achse: Winkel-Bins
Y-Achse: NMPJPE
3 Boxplots pro Bin (einer pro Modell)
```

### 4. Beispielbilder
```
Qualitative Vergleiche bei:
- 0° (frontal)
- 45° (diagonal)
- 90° (seitlich)

Zeigt: GT Skeleton, MediaPipe, MoveNet, YOLO
```

---

## Statistische Analyse (Geplant)

### Tests
1. **Shapiro-Wilk:** Normalverteilung prüfen
2. **Levene:** Varianz-Homogenität
3. **ANOVA:** Unterschied zwischen Modellen pro Bin
4. **Tukey HSD:** Post-hoc paarweise Vergleiche
5. **Regression:** Modellierung des Fehleranstiegs

### Signifikanz-Niveau
α = 0.05

### Effektstärke
Cohen's d für paarweise Vergleiche

---

## Interpretation (Geplant)

### Fragen zu beantworten
1. **Welches Modell ist am robustesten bei Rotation?**
2. **Ab welchem Winkel wird welches Modell unzuverlässig?**
3. **Welche Joints sind am anfälligsten?**
4. **Gibt es einen kritischen Winkel θ_crit?**

### Hypothesen-Überprüfung
- H1: NMPJPE steigt mit Rotation → ?
- H2: Nicht-linearer Anstieg ab ~45° → ?
- H3: Extremitäten > Torso Fehler → ?
- H4: Architektur-Unterschiede → ?
- H5: Kritischer Winkel existiert → ?

---

## Rohdaten-Speicherort

```
data/predictions/
├── Ex1/
│   ├── PM_000-c17.npz
│   │   ├── pred_MediaPipe_heavy
│   │   ├── pred_MoveNet_multipose
│   │   ├── pred_YOLOv8-Pose_n
│   │   ├── rotation_angles
│   │   └── num_frames
│   └── ...
├── Ex2/
└── ...
```

### Laden der Ergebnisse
```python
import numpy as np

data = np.load('data/predictions/Ex1/PM_000-c17.npz')
mediapipe_pred = data['pred_MediaPipe_heavy']
movenet_pred = data['pred_MoveNet_multipose']
yolo_pred = data['pred_YOLOv8-Pose_n']
angles = data['rotation_angles']
```
