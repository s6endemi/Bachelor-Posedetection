# Methodology: Technische Details

Dieses Dokument beschreibt die technischen Details der Implementierung und dient als Grundlage für das Methodology-Kapitel der Thesis.

---

## 1. Rotationswinkel-Berechnung

### Konzept
Der Rotationswinkel beschreibt, wie stark eine Person zur Kamera gedreht steht:
- **0°** = Frontal (Schultern parallel zur Kamera)
- **45°** = Diagonal
- **90°** = Seitlich (eine Schulter verdeckt die andere)

### Berechnung aus 3D Ground Truth
```python
# 3D Schulter-Koordinaten aus Motion Capture
schulter_links = gt_3d[joint_idx_left_shoulder]   # [x, y, z]
schulter_rechts = gt_3d[joint_idx_right_shoulder]  # [x, y, z]

# Differenz in X (Links-Rechts) und Z (Tiefe)
dx = schulter_rechts[0] - schulter_links[0]
dz = schulter_rechts[2] - schulter_links[2]

# Winkel berechnen (0° = frontal)
rotation_rad = np.arctan2(np.abs(dz), np.abs(dx))
rotation_deg = np.degrees(rotation_rad)
```

### Warum aus Ground Truth?
- **Zirkelschluss vermeiden:** Wenn wir den Winkel aus Predictions berechnen würden, wäre die Fehleranalyse verfälscht
- **Genauigkeit:** Motion Capture liefert präzise 3D-Koordinaten
- **Konsistenz:** Alle Modelle werden am gleichen "wahren" Winkel gemessen

### Winkel-Bins
```
Bin 0:  0° - 10°   (fast frontal)
Bin 1: 10° - 20°
Bin 2: 20° - 30°
Bin 3: 30° - 40°
Bin 4: 40° - 50°
Bin 5: 50° - 60°
Bin 6: 60° - 70°
Bin 7: 70° - 80°
Bin 8: 80° - 90°  (fast seitlich)
```

---

## 2. Fehlermetrik: NMPJPE

### Definition
**Normalized Mean Per Joint Position Error**

```python
def calculate_nmpjpe(pred_keypoints, gt_keypoints):
    """
    Args:
        pred_keypoints: Array (N, 2) - Predicted x,y für N Joints
        gt_keypoints: Array (N, 2) - Ground Truth x,y für N Joints

    Returns:
        float: NMPJPE in Prozent der Torso-Länge
    """
    # 1. Euklidische Distanz pro Joint
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)

    # 2. Torso-Länge berechnen (Normalisierungsfaktor)
    shoulder_mid = (gt_keypoints[5] + gt_keypoints[6]) / 2  # L/R Schulter
    hip_mid = (gt_keypoints[11] + gt_keypoints[12]) / 2      # L/R Hüfte
    torso_length = np.linalg.norm(shoulder_mid - hip_mid)

    # 3. Normalisieren und mitteln
    nmpjpe = np.mean(distances) / torso_length * 100

    return nmpjpe
```

### Interpretation
| NMPJPE | Bedeutung | Beispiel (Torso ~120px) |
|--------|-----------|------------------------|
| 5% | Exzellent | ~6px Fehler |
| 10% | Gut | ~12px Fehler |
| 20% | Akzeptabel | ~24px Fehler |
| 50% | Schlecht | ~60px Fehler |
| >100% | Sehr schlecht / Falsche Person | >120px Fehler |

### Warum Normalisierung?
- **Vergleichbarkeit:** Unabhängig von Bildauflösung und Persongröße
- **Interpretierbar:** "10% der Torso-Länge" ist anschaulich
- **Standard:** NMPJPE ist etablierte Metrik in der Literatur

---

## 3. Keypoint-Mapping

### Das Problem
Verschiedene Modelle verwenden verschiedene Keypoint-Formate:
- **MediaPipe:** 33 Keypoints (inkl. Gesicht, Hände, Füße im Detail)
- **MoveNet/YOLO:** 17 COCO Keypoints
- **Ground Truth:** 26 Motion Capture Joints

### Lösung: Gemeinsame Schnittmenge

```python
# 12 vergleichbare Keypoints (alle Modelle + GT)
COMPARABLE_KEYPOINTS = {
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}

# NICHT vergleichbar (fehlen im GT):
# - Nase, Augen, Ohren (Gesicht nicht im Motion Capture)
```

### Mapping-Tabelle

| Body Part | COCO Index | MediaPipe Index | GT Joint |
|-----------|------------|-----------------|----------|
| Left Shoulder | 5 | 11 | LeftArm |
| Right Shoulder | 6 | 12 | RightArm |
| Left Elbow | 7 | 13 | LeftForeArm |
| Right Elbow | 8 | 14 | RightForeArm |
| Left Wrist | 9 | 15 | LeftHand |
| Right Wrist | 10 | 16 | RightHand |
| Left Hip | 11 | 23 | LeftUpLeg |
| Right Hip | 12 | 24 | RightUpLeg |
| Left Knee | 13 | 25 | LeftLeg |
| Right Knee | 14 | 26 | RightLeg |
| Left Ankle | 15 | 27 | LeftFoot |
| Right Ankle | 16 | 28 | RightFoot |

---

## 4. Person-Selection Strategien

### Das Problem
In vielen Frames sind **mehrere Personen** sichtbar (Hintergrund, Therapeut, etc.). Die Modelle müssen die **Hauptperson** (die mit Ground Truth) auswählen.

### Kritische Erkenntnis
> **Verschiedene Modelle brauchen verschiedene Selection-Strategien!**

### Strategie-Übersicht

| Modell | Was es liefert | Selection-Strategie |
|--------|----------------|---------------------|
| **YOLO** | Echte Bounding Box vom Detektor | BBox Area (größte Box) |
| **MoveNet MultiPose** | Echte Bounding Box vom Detektor | BBox Area (größte Box) |
| **MediaPipe** | Nur 33 Keypoints (keine BBox) | **Torso-Größe** |

### Warum BBox bei MediaPipe NICHT funktioniert

**Getestet und dokumentiert:**

| Methode | Fehler bei Multi-Person |
|---------|------------------------|
| BBox aus 33 Keypoints | 11 Fehler |
| BBox aus 12 Body-Keypoints | Immer noch Fehler |
| **Torso-Größe** | **2 Fehler** |

**Erklärung:**
```
BBox misst:   SPREAD (wie weit sich Person ausstreckt)
              → Arme gestreckt = große BBox, egal wie weit weg

Torso misst:  SIZE (wie nah Person zur Kamera ist)
              → Nähere Person = größerer Torso (immer!)
```

### Implementierung: Torso-Größe (MediaPipe)
```python
# Bei mehreren Personen: wähle die größte (Torso-Größe)
if len(results.pose_landmarks) > 1:
    best_idx = 0
    best_size = 0
    for i, lm in enumerate(results.pose_landmarks):
        # Torso-Größe: Schulter-Mitte zu Hüfte-Mitte
        shoulder_y = (lm[11].y + lm[12].y) / 2
        hip_y = (lm[23].y + lm[24].y) / 2
        size = abs(hip_y - shoulder_y)
        if size > best_size:
            best_size = size
            best_idx = i
    landmarks = results.pose_landmarks[best_idx]
```

### Implementierung: BBox Area (YOLO/MoveNet)
```python
# Bei mehreren Personen: wähle die größte Bounding Box
if num_persons > 1 and results[0].boxes is not None:
    best_idx = 0
    best_area = 0
    for i in range(num_persons):
        box = boxes.xyxy[i]  # [x1, y1, x2, y2]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > best_area:
            best_area = area
            best_idx = i
    person_idx = best_idx
```

---

## 5. Ausreißer-Handling

### Definitionen

| Typ | Definition | Ursache |
|-----|------------|---------|
| **Detection Failure** | Keypoints = (0, 0) | Modell erkennt keine Person |
| **Wrong Person** | Schulter >200px von GT | Falsche Person ausgewählt |
| **High Error** | NMPJPE >50% | Schlechte Prediction |

### Handling in der Evaluation
```python
def is_valid_prediction(pred, gt):
    # 1. Detection Failure?
    if pred[0] == 0 and pred[1] == 0:
        return False, "detection_failure"

    # 2. Wrong Person? (Schulter-Check)
    shoulder_dist = np.linalg.norm(pred[5] - gt[5])  # Left shoulder
    if shoulder_dist > 200:  # Pixel
        return False, "wrong_person"

    return True, "valid"
```

### Reporting
Für die Thesis werden alle Ausreißer dokumentiert:
```
| Modell    | Failures | Wrong Person | Valid Frames | % Valid |
|-----------|----------|--------------|--------------|---------|
| MediaPipe |    X     |      X       |      X       |   X%    |
| MoveNet   |    X     |      X       |      X       |   X%    |
| YOLO      |    X     |      X       |      X       |   X%    |
```

**Wichtig:** NMPJPE wird nur auf **valid frames** berechnet!

---

## 6. Modell-Konfigurationen

### MediaPipe
```python
MediaPipeEstimator(
    model_complexity=2,           # Heavy (beste Genauigkeit)
    min_detection_confidence=0.1  # Niedrig (robust)
)
```
- **num_poses=5:** Erlaubt Multi-Person Detection
- **Selection:** Torso-Größe

### MoveNet MultiPose
```python
MoveNetMultiPoseEstimator()
# Model: tfhub.dev/google/movenet/multipose/lightning/1
# Input: 256x256
# Output: 6 Personen × (17 Keypoints × 3 + BBox)
```
- **Selection:** BBox Area
- **Threshold:** score > 0.1

### YOLOv8-Pose
```python
YOLOPoseEstimator(model_size="n")  # oder "m" für Full-Run
# Model: yolov8n-pose.pt / yolov8m-pose.pt
# Input: 640x640 (automatisch skaliert)
```
- **Selection:** BBox Area
- **Verfügbare Größen:** n, s, m, l, x

---

## 7. Pipeline-Ablauf

```
1. Video laden
   ↓
2. Frame extrahieren
   ↓
3. Parallel: 3 Modelle Inference
   ├── MediaPipe → 33 Keypoints → Map to COCO → Select by Torso
   ├── MoveNet → 6×17 Keypoints → Select by BBox
   └── YOLO → N×17 Keypoints → Select by BBox
   ↓
4. Ground Truth laden (2D + 3D)
   ↓
5. Rotationswinkel aus 3D berechnen
   ↓
6. Predictions + Winkel speichern (.npz)
   ↓
7. Evaluation: NMPJPE pro Winkel-Bin
```

### Output-Format (.npz)
```python
{
    'pred_MediaPipe_heavy': (N, 17, 2),      # Predictions
    'pred_MoveNet_multipose': (N, 17, 2),
    'pred_YOLOv8-Pose_n': (N, 17, 2),
    'rotation_angles': (N,),                  # Winkel pro Frame
    'num_frames': int
}
```
