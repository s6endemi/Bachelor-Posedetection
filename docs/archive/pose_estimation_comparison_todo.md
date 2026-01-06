# Pose Estimation Accuracy Comparison: MediaPipe vs MoveNet

## Projektübersicht

Vergleich der Genauigkeit von MediaPipe und MoveNet Pose Detection mit Ground Truth Daten aus dem REHAB24-6 Dataset. Besonderer Fokus: Wie verändert sich die Genauigkeit bei unterschiedlichen Körperdrehungen der Testperson relativ zur Kamera.

**Dataset:** [REHAB24-6](https://zenodo.org/records/13305826)
- 65 Recordings, 184.825 Frames @ 30 FPS
- Ground Truth: 26 Skeleton Joints (3D + 2D projiziert)
- RGB Videos von 2 Kameras

---

## Phase 1: Setup & Datenaufbereitung

### 1.1 Dataset herunterladen und strukturieren
- [ ] `videos.zip` herunterladen und entpacken
- [ ] `2d_joints.zip` herunterladen und entpacken
- [ ] `3d_joints.zip` herunterladen und entpacken
- [ ] `joints_names.txt` herunterladen
- [ ] Ordnerstruktur für Projekt anlegen:
  ```
  project/
  ├── data/
  │   ├── videos/
  │   ├── gt_2d/
  │   ├── gt_3d/
  │   └── predictions/
  │       ├── mediapipe/
  │       └── movenet/
  ├── src/
  ├── results/
  └── visualizations/
  ```
- [ ] Datenformat der GT-Dateien analysieren und dokumentieren (CSV? JSON? Pro Frame?)

### 1.2 Keypoint-Mapping erstellen
- [ ] GT Joint-Namen aus `joints_names.txt` extrahieren und dokumentieren
- [ ] MediaPipe Pose Keypoints dokumentieren (33 Keypoints)
- [ ] MoveNet Keypoints dokumentieren (17 Keypoints)
- [ ] Mapping-Tabelle erstellen: GT ↔ MediaPipe ↔ MoveNet
- [ ] Gemeinsame Keypoints identifizieren (Schnittmenge aller drei)
- [ ] Mapping als Config-Datei speichern (z.B. `keypoint_mapping.json`)

**Erwartete gemeinsame Keypoints (typischerweise):**
- Schultern (links/rechts)
- Ellbogen (links/rechts)
- Handgelenke (links/rechts)
- Hüften (links/rechts)
- Knie (links/rechts)
- Knöchel (links/rechts)
- ggf. Nase, Augen, Ohren

---

## Phase 2: Inference

### 2.1 MediaPipe Pose Detection
- [ ] MediaPipe Pose installieren (`pip install mediapipe`)
- [ ] Inference-Script schreiben:
  - Input: Video-Frames
  - Output: 2D Keypoints + Depth (falls verfügbar) pro Frame
  - Format: Konsistent mit GT-Format für einfachen Vergleich
- [ ] Über alle Videos laufen lassen
- [ ] Ergebnisse speichern in `data/predictions/mediapipe/`
- [ ] Confidence Scores mit abspeichern

### 2.2 MoveNet Pose Detection
- [ ] MoveNet installieren (TensorFlow Hub)
- [ ] Modellvariante wählen (Thunder = genauer, Lightning = schneller)
- [ ] Inference-Script schreiben (analog zu MediaPipe)
- [ ] Über alle Videos laufen lassen
- [ ] Ergebnisse speichern in `data/predictions/movenet/`
- [ ] Confidence Scores mit abspeichern

### 2.3 Datenvalidierung
- [ ] Stichprobenartig prüfen: Stimmt Frame-Anzahl überein (GT vs. Predictions)?
- [ ] Visualisierung einzelner Frames: GT + Predictions überlagert

---

## Phase 3: Koordinaten-Alignment

> **⚠️ HINWEIS:** Diese Phase ist möglicherweise nicht notwendig. Zuerst muss analysiert werden, ob und wie sich die Koordinatensysteme von GT, MediaPipe und MoveNet unterscheiden. Dazu:
> - [ ] Stichprobe von Frames visuell vergleichen: Liegen die Koordinaten bereits übereinander?
> - [ ] Koordinatenbereiche prüfen: Sind GT und Predictions im gleichen Wertebereich (Pixel vs. normalisiert)?
> - [ ] Falls ja → Phase 3 überspringen oder nur minimales Alignment
> - [ ] Falls nein → Alignment wie unten beschrieben durchführen

### 3.1 2D-Koordinaten Alignment
- [ ] Problem analysieren: GT in Pixel-Koordinaten, Predictions evtl. normalisiert
- [ ] Scaling-Algorithmus implementieren:
  - Ansatz: Finde Skalierungsfaktoren (sx, sy) die Gesamtfehler minimieren
  - Methode: Least Squares oder Grid Search
  - Formel: `pred_aligned = pred * [sx, sy] + [tx, ty]` (optional Translation)
- [ ] Pro Video oder global? → Testen was besser funktioniert
- [ ] Alignment validieren und dokumentieren

### 3.2 3D-Koordinaten Alignment (Depth)
- [ ] Analysieren: Welches Depth-Format liefern MediaPipe/MoveNet?
- [ ] GT 3D-Koordinatensystem verstehen (virtuelle cm laut Dataset)
- [ ] Scaling-Strategie festlegen:
  - Option A: Separates Scaling pro Achse (sx, sy, sz)
  - Option B: Uniformes Scaling + Translation
- [ ] Algorithmus implementieren (Minimierung des 3D-Fehlers)
- [ ] Alignment validieren

### 3.3 Alignment-Parameter speichern
- [ ] Optimale Parameter dokumentieren
- [ ] Als Config speichern für Reproduzierbarkeit

---

## Phase 4: Winkelberechnung & Annotation

### 4.1 Rotationswinkel berechnen
- [ ] Funktion implementieren:
  ```python
  def calculate_rotation_angle(left_shoulder_3d, right_shoulder_3d):
      """
      Berechnet den Rotationswinkel der Person zur Kamera.
      0° = Person steht frontal (beide Schultern gleich weit von Kamera)
      Winkel = arctan2(delta_y, delta_x) des Schultervektors
      """
  ```
- [ ] Winkel für jeden Frame berechnen (aus GT 3D-Daten)
- [ ] Winkelverteilung im Dataset analysieren und visualisieren

### 4.2 Frames mit Winkel annotieren
- [ ] Winkel-Annotation zu jedem Frame hinzufügen
- [ ] Datenstruktur erweitern:
  ```
  frame_data = {
      "frame_id": ...,
      "rotation_angle": ...,  # in Grad
      "gt_keypoints_2d": ...,
      "gt_keypoints_3d": ...,
      "mediapipe_keypoints": ...,
      "movenet_keypoints": ...
  }
  ```
- [ ] Annotierte Daten speichern

### 4.3 Winkel-Bins definieren
- [ ] Bin-Größe: 3° (z.B. 0.0-2.9°, 3.0-5.9°, 6.0-8.9°, ...)
- [ ] Bins bis zum maximalen vorkommenden Winkel erstellen
- [ ] Frames den Bins zuordnen
- [ ] Anzahl Frames pro Bin dokumentieren (für statistische Signifikanz)

---

## Phase 5: Fehlerberechnung & Auswertung

### 5.1 Per-Joint Error Metriken implementieren
- [ ] 2D Euclidean Distance pro Joint:
  ```python
  error_2d = sqrt((gt_x - pred_x)² + (gt_y - pred_y)²)
  ```
- [ ] 3D Euclidean Distance pro Joint (falls Depth verfügbar):
  ```python
  error_3d = sqrt((gt_x - pred_x)² + (gt_y - pred_y)² + (gt_z - pred_z)²)
  ```
- [ ] Normalisierung überlegen (z.B. durch Körpergröße/Torso-Länge)

### 5.2 Fehler berechnen
- [ ] Per-Joint Error für jeden Frame berechnen
- [ ] Für beide Modelle (MediaPipe, MoveNet)
- [ ] Für beide Error-Typen (2D, 3D)
- [ ] Ergebnisse strukturiert speichern

### 5.3 Aggregation nach Winkel-Bins
- [ ] Pro Winkel-Bin berechnen:
  - Mean Error (pro Joint)
  - Std Error (pro Joint)
  - Mean Error (über alle Joints gemittelt)
  - Median Error
  - Anzahl Frames im Bin
- [ ] Ergebnisse als Tabelle speichern

### 5.4 Statistische Auswertung
- [ ] Signifikanz der Unterschiede zwischen Bins testen
- [ ] Vergleich MediaPipe vs MoveNet pro Winkel-Bin
- [ ] Identifizieren: Bei welchen Winkeln degradiert die Performance?
- [ ] Identifizieren: Welche Joints sind besonders betroffen?

---

## Phase 6: Visualisierung

### 6.1 Hauptgrafik: Error vs. Rotationswinkel
- [ ] X-Achse: Rotationswinkel (in 3°-Bins)
- [ ] Y-Achse: Mean Per-Joint Error
- [ ] Zwei Linien: MediaPipe, MoveNet
- [ ] Error-Bars für Standardabweichung
- [ ] Gut lesbare Achsenbeschriftungen und Legende

### 6.2 Weitere Visualisierungen
- [ ] Heatmap: Error pro Joint und Winkel-Bin
- [ ] Boxplots: Error-Verteilung pro Winkel-Bin
- [ ] Histogramm: Winkelverteilung im Dataset
- [ ] Beispiel-Frames: Overlay GT vs. Predictions bei verschiedenen Winkeln

### 6.3 Export
- [ ] Grafiken in hoher Auflösung speichern (PNG, PDF)
- [ ] Rohdaten für Grafiken als CSV exportieren

---

## Phase 7: Dokumentation & Abschluss

### 7.1 Ergebnisse dokumentieren
- [ ] Zusammenfassung der Haupterkenntnisse
- [ ] Limitationen der Analyse
- [ ] Vergleich mit anderen Studien (falls vorhanden)

---

## Offene Fragen / Notizen

- Wie genau funktioniert das Depth-Output von MediaPipe/MoveNet? → Recherchieren
- Sollen nur Frames mit hoher Confidence (>0.5?) verwendet werden?
- Wie mit fehlenden Detections umgehen (Person nicht erkannt)?
- Beide Kameras (horizontal/vertikal) separat auswerten oder zusammen?

---

## Abhängigkeiten

```
python >= 3.8
mediapipe
tensorflow / tensorflow-hub (für MoveNet)
numpy
pandas
matplotlib
scipy (für Optimierung/Statistik)
opencv-python
```
