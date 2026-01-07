# Problems & Solutions: Die Journey

Dieses Dokument ist das **Herzstück der Dokumentation**. Hier werden alle Probleme, deren Ursachen und Lösungen festgehalten. Diese Erkenntnisse sind essentiell für das Discussion-Kapitel der Thesis.

---

## Problem 1: MediaPipe Detection Failures (29%)

### Symptom
Bei ersten Tests hatte MediaPipe eine **Detection Failure Rate von 29%** - fast ein Drittel aller Frames!

### Beobachtung
```
Frame 50: MediaPipe gibt (0, 0, 0, ...) zurück
Frame 51: Normale Keypoints
Frame 52: Wieder (0, 0, 0, ...)
```

### Root Cause Analyse
Der Default-Wert `min_detection_confidence=0.5` war **zu strikt**.

**Was passiert bei confidence=0.5:**
1. MediaPipe detectiert Person mit confidence=0.45
2. 0.45 < 0.5 → Detection wird verworfen
3. Modell gibt leere Keypoints zurück → (0,0)
4. Diese werden als "Detection Failure" gezählt

### Kontraintuitiv!
> **Höhere Confidence = Mehr Fehler, nicht weniger!**

Das ist kontraintuitiv, aber logisch:
- Confidence ist ein **Filter**, kein Qualitätsmaß
- Ein zu strenger Filter verwirft auch gute Detections
- Besser: Niedrige Confidence, dann Quality-Check nachgelagert

### Lösung
```python
# Vorher
min_detection_confidence=0.5  # Default - ZU STRIKT

# Nachher
min_detection_confidence=0.1  # Niedrig, robust
```

### Ergebnis
| Metrik | Vorher | Nachher |
|--------|--------|---------|
| Detection Failures | 29% | 0.4% |
| NMPJPE | 128% (!) | ~10% |

### Lesson Learned
> **Defaults immer hinterfragen!** MediaPipe's Default ist für "saubere" Szenarien optimiert, nicht für Real-World mit Bewegung und Rotation.

---

## Problem 2: YOLO wählt falsche Person (37% NMPJPE)

### Symptom
Bei 50-60° Rotation hatte YOLO plötzlich **37% NMPJPE** - deutlich schlechter als bei anderen Winkeln.

### Erste Vermutung (FALSCH!)
"YOLO hat Probleme mit Rotation bei 50-60°"

### Tatsächliche Ursache
Eine **Frau lief während der 50-60° Frames durch das Bild**. YOLO wählte sie statt der Hauptperson!

### Debug-Prozess
1. Debug-Bild mit Predictions gespeichert
2. Gesehen: YOLO-Skeleton war ~600px von GT entfernt
3. Video geschaut: Frau im Hintergrund
4. Code analysiert: `keypoints.xy[0]` - nimmt ERSTE Person, nicht GRÖSSTE

### Root Cause
```python
# FALSCH: Erste Person nehmen
person_keypoints = keypoints.xy[0]

# Kein Check ob das die richtige Person ist!
```

### Lösung: BBox Area Selection
```python
# RICHTIG: Größte Bounding Box wählen
if num_persons > 1 and results[0].boxes is not None:
    best_idx = 0
    best_area = 0
    for i in range(num_persons):
        box = boxes.xyxy[i]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > best_area:
            best_area = area
            best_idx = i
    person_idx = best_idx
```

### Ergebnis
| Metrik | Vorher | Nachher |
|--------|--------|---------|
| NMPJPE bei 50-60° | 37% | ~14% |
| Wrong Person | Häufig | 0% |

### Lesson Learned
> **Korrelation ≠ Kausalität!** Der hohe Fehler bei 50-60° war KEINE Rotationsschwäche, sondern ein Person-Selection Bug.

---

## Problem 3: MoveNet SinglePose Limitation

### Symptom
MoveNet hatte bei 40-50° plötzlich **32% NMPJPE** mit riesiger Standardabweichung.

### Debug
1. Debug-Bild gespeichert
2. MoveNet-Skeleton war auf **Hintergrund-Person** (Frau)
3. Aber: MoveNet SinglePose kann nur **1 Person** detecten!

### Root Cause
**Architektur-Limitation:** SinglePose hat keine Multi-Person Fähigkeit.

```
SinglePose:
- Input: Bild
- Output: 1 × 17 Keypoints
- Problem: KEINE Auswahl möglich!

Wenn Hintergrund-Person "dominanter" ist → falsche Person
```

### Lösungsoptionen
1. ❌ **BBox Selection:** Geht nicht (nur 1 Person)
2. ❌ **Confidence Filter:** Hilft nicht bei falscher Person
3. ✅ **Modell wechseln:** MultiPose verwenden

### Lösung: MoveNet MultiPose
```python
# Vorher: SinglePose (kann nicht selectieren)
url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"

# Nachher: MultiPose (bis zu 6 Personen)
url = "https://tfhub.dev/google/movenet/multipose/lightning/1"
```

**MultiPose Output-Format:**
```python
# Shape: (1, 6, 56)
# Pro Person: 17 keypoints × 3 (x,y,score) + bbox (4) + person_score (1)
# = 51 + 4 + 1 = 56

# BBox ist in indices [51:55]: ymin, xmin, ymax, xmax
```

### Implementierung
Neuer Estimator `MoveNetMultiPoseEstimator`:
- Lädt MultiPose Modell
- Wählt Person mit größter BBox Area
- Mapped auf COCO Format

### Ergebnis
| Metrik | SinglePose | MultiPose |
|--------|------------|-----------|
| Wrong Person | 2.8% | 0% |
| Architektur | Keine Auswahl | BBox Selection |

### Lesson Learned
> **Modell-Architektur verstehen!** SinglePose vs MultiPose ist fundamental - man kann Selection nicht nachträglich hinzufügen.

### Für die Thesis
Wir dokumentieren **beide** Varianten:
- SinglePose: Als Limitation dokumentieren
- MultiPose: Als Lösung und finales Modell

---

## Problem 4: BBox funktioniert nicht für MediaPipe

### Symptom
Nach den YOLO/MoveNet Fixes wollten wir auch MediaPipe auf BBox umstellen. Ergebnis: **Mehr Fehler!**

### Experiment
```python
# Test: 100 Frames mit Multi-Person

# Methode 1: BBox aus allen 33 Keypoints
bbox_all_errors = 11

# Methode 2: BBox aus nur 12 Body-Keypoints
bbox_body_errors = 8  # Immer noch schlecht

# Methode 3: Torso-Größe (Schulter-Hüfte Distanz)
torso_errors = 2  # BESTE
```

### Root Cause
MediaPipe liefert **keine echten Bounding Boxes**, nur Keypoints.

**Das Problem mit Pseudo-BBox:**
```
BBox aus Keypoints = min/max aller Koordinaten

Problem: Misst SPREAD (Ausdehnung), nicht SIZE (Größe)

Beispiel:
- Hauptperson: Arme eng am Körper → kleine BBox
- Hintergrund: Arme gestreckt → GROSSE BBox
→ Falsche Person wird gewählt!
```

### Warum Torso funktioniert
```
Torso-Größe = Distanz(Schulter-Mitte, Hüfte-Mitte)

Vorteile:
1. Immer sichtbar (auch bei Occlusion)
2. Korreliert mit Kamera-Distanz
3. Unabhängig von Arm-Position

Nähere Person = Größerer Torso (IMMER)
```

### Vergleich der Ansätze

| Modell | Echte BBox? | Selection-Strategie |
|--------|-------------|---------------------|
| YOLO | ✅ Ja (vom Detektor) | BBox Area |
| MoveNet MP | ✅ Ja (vom Detektor) | BBox Area |
| MediaPipe | ❌ Nein (nur Keypoints) | Torso-Größe |

### Lesson Learned
> **Eine Lösung passt nicht für alle!** Verschiedene Modelle brauchen verschiedene Selection-Strategien basierend auf ihren Outputs.

### Für die Thesis
Wichtige methodische Erkenntnis:
> "Die Person-Selektion wurde modell-spezifisch implementiert: YOLO und MoveNet liefern Bounding Boxes aus dem Detektions-Netzwerk, daher wurde die Box-Fläche als Kriterium verwendet. MediaPipe liefert keine Bounding Boxes, daher wurde die Torso-Größe (Schulter-Hüfte-Distanz) verwendet, welche besser mit der Kamera-Distanz korreliert."

---

## Problem 5: Erste Torso-Selection für YOLO (falscher Ansatz)

### Kontext
Nach dem YOLO-Bug haben wir zuerst Torso-Größe als Selection versucht (wie bei MediaPipe).

### Ergebnis
Funktionierte **nicht zuverlässig** - immer noch falsche Personen.

### Warum?
Bei YOLO sind die Torso-Größen manchmal ähnlich, aber die **BBox ist deutlich unterschiedlich**.

```
Hauptperson:  Torso=0.18, BBox=50000 px²
Hintergrund:  Torso=0.15, BBox=15000 px²

→ Torso-Differenz: 20%
→ BBox-Differenz: 230%

BBox ist ein stärkeres Signal!
```

### Lesson Learned
> **Wenn echte BBox verfügbar, nutze sie!** Pseudo-Metriken (Torso aus Keypoints) nur als Fallback.

---

## Zusammenfassung: Selection-Matrix

| Modell | Hat echte BBox? | Selection | Warum? |
|--------|-----------------|-----------|--------|
| YOLO | ✅ | BBox Area | Stärkstes Signal vom Detektor |
| MoveNet MP | ✅ | BBox Area | Stärkstes Signal vom Detektor |
| MediaPipe | ❌ | Torso-Größe | Korreliert mit Kamera-Distanz |
| MoveNet SP | ❌ (nur 1 Person) | - | Nicht verwendbar für Multi-Person |

---

## Meta-Lessons für die Thesis

### 1. Debugging-Strategie
- **Immer visuell validieren** - Debug-Bilder zeigen Probleme sofort
- **Mini-Test vor Full-Run** - Spart Stunden bei Bugs
- **Korrelation ≠ Kausalität** - Rotationsfehler war Person-Verwechslung

### 2. Modell-Verständnis
- **Architektur-Unterschiede sind fundamental** - SinglePose vs MultiPose
- **Outputs verstehen** - Was liefert das Modell wirklich?
- **Defaults hinterfragen** - confidence=0.5 war zu strikt

### 3. Methodische Sauberkeit
- **Nicht blind kopieren** - Eine Lösung passt nicht für alle
- **Modell-spezifische Lösungen** - Torso für MediaPipe, BBox für YOLO
- **Dokumentieren während der Arbeit** - Nicht erst am Ende

### 4. Real-World Challenges
- **Multi-Person ist normal** - Hintergrundpersonen kommen vor
- **Natürliche Szenarien sind härter** - COCO Benchmark ist "zu sauber"
- **Edge Cases finden** - Rotation + Multi-Person = Probleme

---

## Timeline der Entdeckungen

```
Tag 1: MediaPipe 128% NMPJPE
       → Debug → confidence=0.5 Problem
       → Fix: confidence=0.1

Tag 2: YOLO 37% bei 50-60°
       → Debug → Falsche Person
       → Fix: BBox Selection

Tag 2: MoveNet 32% bei 40-50°
       → Debug → SinglePose Limitation
       → Fix: MultiPose Variante

Tag 3: MediaPipe BBox Test
       → Experiment → Macht es schlimmer
       → Erkenntnis: Torso > BBox für Keypoint-only

Tag 3: Pipeline validiert
       → Mini-Test: 500 Frames
       → Alle Modelle: <1% Fehler
```

---

## Erkenntnis 6: Motion Capture Rotationswinkel-Variation

### Beobachtung
Bei Video PM_000-c17 variiert der Rotationswinkel zwischen **45° und 55°**, obwohl die Person augenscheinlich still steht und nur die Arme hebt/senkt.

### Erste Vermutung
"Bug in unserer Rotationswinkel-Berechnung?"

### Analyse

**Unsere Formel:**
```python
θ = arctan2(|dz|, |dx|)

# Mit:
dz = left_shoulder_z - right_shoulder_z
dx = left_shoulder_x - right_shoulder_x
```

**Untersuchung der 3D Ground Truth:**
```python
# Video PM_000: 1304 Frames, Person steht "still"
# Left Shoulder (Joint 7) z-Koordinate:
Frame    0: z = 0.10
Frame  500: z = 0.20
Frame 1000: z = 0.20

# Resultierende Winkel:
Min: 45.40°  Max: 55.27°  Mean: 49.58°  Std: 2.56°
```

### Root Cause: Dataset, NICHT Berechnung

Die Variation kommt aus dem **Motion Capture System**, nicht aus unserer Formel:

1. **Armbewegungen bewegen Schultermarker**
   - Arm heben → Schultergelenk rotiert → Marker verschiebt sich
   - ~10cm Verschiebung in z-Richtung beobachtet

2. **MoCap-Systemrauschen**
   - Optische Systeme haben ±1-2cm Genauigkeit
   - Marker-Okklusion bei Bewegung

3. **Körperschwankungen**
   - Niemand steht 100% still
   - Natürliche Gewichtsverlagerung

### Mathematische Erklärung der Sensitivität

Bei ~45° gilt: `|dz| ≈ |dx|`

Kleine Änderungen haben dann überproportionalen Effekt:
```
arctan2(0.10, 0.15) = 33.7°
arctan2(0.20, 0.15) = 53.1°
→ +0.10 in z führt zu +20° Änderung!
```

Bei 0° oder 90° wäre die Sensitivität viel geringer.

### Validierung: Kein Bug

| Check | Ergebnis |
|-------|----------|
| Formel korrekt? | ✅ arctan2 ist Standard |
| Deterministisch? | ✅ Gleiche Inputs = gleiche Outputs |
| GT-Daten variieren? | ✅ Ja, Schulter-z ändert sich |
| Variation erklärbar? | ✅ Armbewegung + MoCap-Noise |

### Implikation für Methodik

- **Std = 2.56°** ist akzeptabel für **10°-Bins**
- Die Variation ist **physikalische Realität**, kein Artefakt
- Binning-Strategie absorbiert natürliche Schwankungen

### Lesson Learned
> **Ground Truth ist nicht perfekt!** Auch Motion Capture Daten haben Rauschen und bewegungsbedingte Variationen. Die Wahl von 10°-Bins ist methodisch sinnvoll, um diese natürlichen Schwankungen zu absorbieren.

### Für die Thesis
> "Der Rotationswinkel wurde aus den 3D Motion Capture Daten berechnet. Eine Analyse zeigte eine Standardabweichung von ~2.5° selbst bei statischen Posen, bedingt durch Marker-Verschiebung bei Armbewegungen und systeminhärentes Rauschen. Die Wahl von 10°-Bins absorbiert diese natürliche Variation."

---

---

## Problem 6: Kamera-Koordinatensystem (KRITISCH)

### Symptom
Die berechneten Rotationswinkel ergaben keinen Sinn:
- PM_002 hatte 0-10 Grad MoCap-Winkel, aber Person stand **seitlich** zur Kamera
- PM_000 hatte 45-55 Grad MoCap-Winkel, aber Person stand **frontal** zur Kamera

### Root Cause
Die Kameras stehen nicht entlang der MoCap-Koordinatenachsen!
- **Camera 17** schaut aus einem Winkel von ~65 Grad zum MoCap-System
- **Camera 18** ist 90 Grad zu Camera 17 gedreht

### Debug-Prozess
1. Videos visuell analysiert: "Wann steht Person frontal?"
2. Referenz-Videos identifiziert: PM_114, PM_122, PM_109
3. MoCap-Winkel bei "visuell frontal" gemessen: ~65 Grad

### Loesung: Empirische Offset-Transformation
```python
# In pipeline.py
C17_FRONTAL_OFFSET = 65.0  # MoCap-Winkel bei frontal zu c17

def calculate_rotation_angle(self, gt_3d_frame, camera):
    # MoCap-Winkel berechnen
    mocap_angle = np.degrees(np.arctan2(abs(dz), abs(dx)))

    # Transformation zu kamera-relativ
    c17_relative = abs(mocap_angle - C17_FRONTAL_OFFSET)

    if camera == 'c17':
        return c17_relative
    else:  # c18 ist 90 Grad gedreht
        return 90.0 - c17_relative
```

### Validierung
| Video | MoCap | c17 kamera-rel | c18 kamera-rel | Beobachtung |
|-------|-------|----------------|----------------|-------------|
| PM_000 | 50 | 15 (frontal) | 75 (seitlich) | Korrekt! |
| PM_002 | 2 | 63 (seitlich) | 27 (schraeg frontal) | Korrekt! |
| PM_114 | 68 | 3 (frontal) | 87 (seitlich) | Korrekt! |

### Erste Ergebnisse nach Fix
| Winkel (kamera-relativ) | NMPJPE |
|------------------------|--------|
| 0-20 (frontal) | 8-11% |
| 70-90 (seitlich) | 15-18% |

**Hypothese bestaetigt: Seitliche Ansichten haben ~2x hoehere Fehler!**

### Lesson Learned
> **Koordinatensysteme IMMER validieren!** MoCap-Koordinaten sind nicht automatisch Kamera-Koordinaten. Die Transformation muss explizit durchgefuehrt werden.

### Fuer die Thesis
> "Die Rotationswinkel wurden aus den 3D Motion Capture Daten berechnet und anschliessend in das kamera-relative Koordinatensystem transformiert. Der Offset wurde empirisch aus Referenzvideos bestimmt, bei denen die Probanden visuell frontal zur jeweiligen Kamera standen (C17_FRONTAL_OFFSET = 65 Grad). Camera 18 ist 90 Grad zu Camera 17 gedreht."

### Warum nicht PnP?
PnP (Perspective-n-Point) waere mathematisch exakter, aber:
1. Keine Kamera-Intrinsics im Dataset
2. Empirischer Ansatz reicht fuer 10-Grad-Bins
3. Ergebnisse wurden visuell validiert

---

## Problem 7: MediaPipe Unterkörper-Ausreißer

### Symptom
Bei ~22% der Frames hatte MediaPipe extrem hohe Fehler (>100px) am Unterkörper, während MoveNet/YOLO stabil blieben.

### Debug-Prozess
1. Verglichen gute vs. schlechte Frames
2. Entdeckt: Bei schlechten Frames hat MediaPipe **niedrige Joint-Confidence** (0.003-0.07)
3. Bei guten Frames: Joint-Confidence > 0.95

### Root Cause
MediaPipe liefert Confidence pro Keypoint. Bei Multi-Person-Szenarien oder schwierigen Posen wird MediaPipe unsicher - und zeigt das durch niedrige Confidence an. Wir haben diese Information ignoriert.

### Loesung: Confidence-Filter
```python
# Nur Joints mit Confidence >= 0.5 in Evaluation einbeziehen
MIN_JOINT_CONFIDENCE = 0.5
```

### Ergebnis
| Modell | Ohne Filter | Mit Filter (0.5) | Gefilterte Joints |
|--------|-------------|------------------|-------------------|
| MediaPipe | 18.3% | 12.9% | 6.8% |
| MoveNet | ~12% | ~12% | 2.2% |
| YOLO | ~12% | ~12% | 0.0% |

### Lesson Learned
> **Confidence-Werte sind nicht optional!** Sie zeigen die Zuverlaessigkeit der Detektion an. Ein Modell das "ich bin unsicher" sagt, sollte man nicht zwingen eine Antwort zu geben.

### Fuer die Thesis
> "Keypoints mit Confidence < 0.5 wurden von der Evaluation ausgeschlossen. Dies betraf 6.8% der MediaPipe-Joints, 2.2% bei MoveNet, und 0% bei YOLO. Die unterschiedlichen Skip-Raten deuten auf unterschiedliche Zuverlaessigkeit der Modelle hin - YOLO ist stets confident, waehrend MediaPipe bei schwierigen Szenarien haeufiger unsicher ist."

---

## Offene Fragen (fuer Full-Run)

1. **Wie verhält sich MoveNet Thunder vs Lightning?**
   - Lightning ist schneller aber weniger genau
   - MultiPose nur als Lightning verfuegbar

2. **Confidence-Thresholds optimieren?**
   - Aktuell: MediaPipe=0.1, MoveNet=0.1, YOLO=default
   - Koennte man tunen, aber nicht primaeres Ziel

3. **Per-Joint Analyse bei hoher Rotation?**
   - Hypothese: Extremitaeten leiden mehr
   - Wird im Full-Run analysiert
