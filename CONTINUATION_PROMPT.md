# Continuation Prompt für nächste AI Session

## Projekt-Kontext
Bachelorarbeit: Vergleich von Pose Estimation Modellen (MediaPipe, MoveNet, YOLO) bei verschiedenen Körper-Rotationswinkeln zur Kamera.

## Wichtige Dateien zum Einlesen
```
ZWISCHENSTAND.md                    # Aktueller Status (ZUERST LESEN!)
docs/02_PROBLEMS_AND_SOLUTIONS.md   # Alle gefundenen Probleme & Lösungen
docs/01_METHODOLOGY.md              # Technische Details
```

## Aktueller Stand (07.01.2026)

### Was funktioniert
- Pipeline komplett implementiert (Inference, Evaluation, Visualisierungen)
- 3 Modelle: MediaPipe Heavy, MoveNet MultiPose, YOLOv8-Pose
- Person-Selection modell-spezifisch (BBox für YOLO/MoveNet, Torso für MediaPipe)
- Evaluator mit NMPJPE-Berechnung pro Winkel-Bin + **Confidence-Filter (0.5)**
- **Kamera-Koordinatensystem GELOEST** (C17_FRONTAL_OFFSET = 65°)
- Ex1 Inference: 9/26 Videos fertig

### GELOEST: Kamera-Koordinatensystem
Empirisch bestimmt: Bei MoCap-Winkel 65° steht Person frontal zu Camera17.

```python
# In pipeline.py
C17_FRONTAL_OFFSET = 65.0
c17_relative = abs(mocap_angle - C17_FRONTAL_OFFSET)
c18_relative = 90.0 - c17_relative
```

### GELOEST: MediaPipe Unterkörper-Ausreißer
MediaPipe gibt niedrige Confidence (0.003-0.07) für unsichere Joints.
Lösung: `MIN_JOINT_CONFIDENCE = 0.5` in evaluator.py - filtert ~15% der Frames.

## Daten-Struktur

```
data/
├── videos/Ex1/              # MP4 Videos
├── gt_2d/Ex1/               # 2D Ground Truth (N, 26, 2) pixel coords
├── gt_3d/Ex1/               # 3D Ground Truth (N, 26, 4) world coords
└── predictions/Ex1/         # Inference Output (.npz)
    └── PM_000-c17.npz       # enthält: pred_*, rotation_angles, num_frames
```

## Code-Struktur

```
run_inference.py                    # Hauptskript
src/pose_evaluation/
├── estimators/                     # MediaPipe, MoveNet, YOLO
├── evaluation/evaluator.py         # NMPJPE Berechnung
└── data/keypoint_mapping.py        # Joint-Mapping (12 gemeinsame)
```

## Keypoint-Mapping

```python
COCO_INDICES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # Predictions
GT_INDICES = [7, 12, 8, 13, 9, 14, 16, 21, 17, 22, 18, 23]  # Ground Truth
# Mapping: Shoulders, Elbows, Wrists, Hips, Knees, Ankles (12 Joints)
```

## Commands

```bash
# Virtual Environment aktivieren
.venv/Scripts/python <script.py>

# Inference (Ex1 läuft vermutlich noch/ist fertig)
.venv/Scripts/python run_inference.py --exercise Ex1

# Evaluation
.venv/Scripts/python -c "
from pathlib import Path
from src.pose_evaluation.evaluation.evaluator import Evaluator
evaluator = Evaluator(predictions_dir=Path('data/predictions/Ex1'), gt_2d_dir=Path('data/gt_2d'))
results = evaluator.evaluate_all()
"
```

## Erstellte Visualisierungen (im Root)
- `evaluation_nmpjpe_explained.png` - NMPJPE vs Rotation + Erklärung
- `comparison_rotation_*.png` - Skeleton-Vergleiche bei verschiedenen Winkeln
- `debug_rotation_check.png` - Zeigt das Koordinatensystem-Problem

## TODO für nächste Session

1. **Ex1 Inference fortsetzen** (9/26 fertig)
   ```bash
   .venv/Scripts/python run_inference.py --exercise Ex1
   # Startet automatisch bei Video 10 dank skip_existing
   ```

2. **Full-Run starten** (alle Exercises)
   ```bash
   .venv/Scripts/python run_inference.py
   ```

3. **Finale Evaluation**
   ```bash
   .venv/Scripts/python -c "
   from pathlib import Path
   from src.pose_evaluation.evaluation.evaluator import Evaluator
   e = Evaluator(predictions_dir=Path('data/predictions'), gt_2d_dir=Path('data/gt_2d'))
   results = e.evaluate_all()
   e.save_results(results, Path('data/predictions/final_results'))
   "
   ```

4. **Statistische Analyse + Plots für Thesis**

## Wichtige Erkenntnisse (dokumentiert in docs/02)
- MediaPipe braucht confidence=0.1 (nicht 0.5!) für Detection
- **Evaluation:** MIN_JOINT_CONFIDENCE = 0.5 um schlechte Joints rauszufiltern
- BBox-Selection für YOLO/MoveNet, Torso für MediaPipe
- MoCap hat ~2.5° Std Variation bei stehender Person
- MoveNet MultiPose statt SinglePose verwenden
- Kamera-Offset: C17_FRONTAL_OFFSET = 65° (empirisch bestimmt)
