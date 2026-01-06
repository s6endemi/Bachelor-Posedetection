# Continuation Prompt für nächste AI Session

## Projekt-Kontext
Bachelorarbeit: Vergleich von Pose Estimation Modellen (MediaPipe, MoveNet, YOLO) bei verschiedenen Körper-Rotationswinkeln zur Kamera.

## Wichtige Dateien zum Einlesen
```
ZWISCHENSTAND.md                    # Aktueller Status (ZUERST LESEN!)
docs/02_PROBLEMS_AND_SOLUTIONS.md   # Alle gefundenen Probleme & Lösungen
docs/01_METHODOLOGY.md              # Technische Details
```

## Aktueller Stand (06.01.2026)

### Was funktioniert
- Pipeline komplett implementiert (Inference, Evaluation, Visualisierungen)
- 3 Modelle: MediaPipe Heavy, MoveNet MultiPose, YOLOv8-Pose
- Person-Selection modell-spezifisch (BBox für YOLO/MoveNet, Torso für MediaPipe)
- Evaluator mit NMPJPE-Berechnung pro Winkel-Bin
- Ex1 Inference läuft im Hintergrund (war bei 10/26 Videos)

### KRITISCHES PROBLEM: Kamera-Koordinatensystem
**Die berechneten Rotationswinkel sind MoCap-relativ, NICHT Kamera-relativ!**

Beobachtung:
- PM_002 hat 0-10° MoCap-Winkel → Person steht SEITLICH zur Kamera
- PM_000 hat 45-55° MoCap-Winkel → Person steht FRONTAL zur Kamera

Die Kameras stehen gedreht zum MoCap-System:
- Camera17: ~frontal wenn MoCap ~45-50°
- Camera18: ~seitlich (90° zu Camera17)

### Nächster Schritt: Kamera-Extrinsics berechnen

**Option 1: PnP (empfohlen)**
```python
import cv2
import numpy as np

# Wir haben:
# - 3D GT: data/gt_3d/Ex1/PM_000-30fps.npy (N, 26, 4) - x,y,z,confidence
# - 2D GT: data/gt_2d/Ex1/PM_000-c17-30fps.npy (N, 26, 2) - pixel x,y

# PnP lösen:
# success, rvec, tvec = cv2.solvePnP(object_points_3d, image_points_2d, camera_matrix, dist_coeffs)
# rvec enthält die Kamera-Rotation

# Dann: Kamera-relativen Winkel berechnen
# θ_camera = θ_mocap - camera_yaw_offset
```

**Benötigt:**
- Kamera-Intrinsics (focal length, principal point) - ggf. schätzen oder aus Metadaten
- Mindestens 4 korrespondierende Punkte (haben wir: 26 Joints!)

**Option 2: Empirisch**
- Aus Videos visuell bestimmen bei welchem MoCap-Winkel Person frontal steht
- Camera17 Offset ≈ 45-50°
- Camera18 Offset ≈ Camera17 + 90°

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

1. **Prüfen ob Ex1 Inference fertig ist**
   ```bash
   dir data\predictions\Ex1 | find /c ".npz"
   # Sollte 26 sein
   ```

2. **Kamera-Extrinsics mit PnP berechnen**
   - Für Camera17 und Camera18 separat
   - Kamera-Yaw-Offset extrahieren
   - Funktion schreiben: `mocap_angle_to_camera_angle(θ_mocap, camera_id)`

3. **Rotation-Berechnung anpassen**
   - Entweder: Post-Processing der gespeicherten Winkel
   - Oder: In Pipeline einbauen für Full-Run

4. **Nach Fix: Full-Run starten**
   ```bash
   .venv/Scripts/python run_inference.py
   ```

5. **Finale Evaluation + Plots**

## Wichtige Erkenntnisse (dokumentiert in docs/02)
- MediaPipe braucht confidence=0.1 (nicht 0.5!)
- BBox-Selection für YOLO/MoveNet, Torso für MediaPipe
- MoCap hat ~2.5° Std Variation bei stehender Person
- MoveNet MultiPose statt SinglePose verwenden
