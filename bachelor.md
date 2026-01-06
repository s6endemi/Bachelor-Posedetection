# Bachelorarbeit: Rotationsabhängige Genauigkeit von 2D Pose Estimation

## Arbeitstitel

**"Einfluss des Körper-Rotationswinkels auf die Genauigkeit von 2D Pose Estimation: Ein systematischer Vergleich von MediaPipe, MoveNet und YOLOv8-Pose"**

---

## 1. Einleitung

### 1.1 Motivation & Problemstellung

- Pose Estimation wird zunehmend in mobilen Gesundheitsanwendungen eingesetzt (Physiotherapie, Fitness-Tracking, Rehabilitation)
- Problem: In der Praxis positionieren sich Nutzer nicht immer optimal frontal zur Kamera
- **Kernproblem:** Es fehlt eine systematische Analyse, wie stark die Genauigkeit bei verschiedenen Rotationswinkeln degradiert
- **Praktische Relevanz:** Für Apps wie Previa Health ist es essenziell zu wissen, ab welchem Winkel die Analyse unzuverlässig wird

### 1.2 Forschungsfrage

> **Hauptfrage:** Wie verändert sich die 2D-Keypoint-Genauigkeit von Pose Estimation Modellen in Abhängigkeit vom Rotationswinkel der Person zur Kamera?

**Unterfragen:**
1. Ab welchem Rotationswinkel degradiert die Genauigkeit signifikant (>X% Fehleranstieg)?
2. Gibt es systematische Unterschiede zwischen verschiedenen Modellarchitekturen (Top-Down, Bottom-Up, One-Stage) bei verschiedenen Winkeln?
3. Welche Körperregionen (Joints) sind besonders anfällig für rotationsbedingte Fehler?
4. Lassen sich praktische Schwellenwerte für mobile Anwendungen ableiten?

### 1.3 Beitrag der Arbeit

1. **Systematisches Evaluationsprotokoll** für winkelabhängige Pose Estimation Analyse
2. **Quantitative Daten** zur Genauigkeitsdegradation bei Rotation
3. **Praktische Guidelines** für Entwickler mobiler Gesundheits-Apps
4. **Architekturvergleich:** Drei fundamentale Ansätze (Top-Down, Bottom-Up, One-Stage) im direkten Vergleich

### 1.4 Abgrenzung

Diese Arbeit fokussiert sich auf:
- **2D Pose Estimation** (keine 3D-Rekonstruktion)
- **Single-Person** Szenarien
- **Statische Rotationswinkel** (nicht dynamische Bewegungen)
- **On-Device Modelle** (MediaPipe, MoveNet, YOLOv8-Pose) die für mobile Apps relevant sind

---

## 2. Related Work

### 2.1 Pose Estimation Grundlagen
- [ ] Überblick: Top-Down vs. Bottom-Up vs. One-Stage Ansätze
- [ ] Convolutional Pose Machines, Stacked Hourglass, HRNet
- [ ] Lightweight Modelle für Mobile: MoveNet, MediaPipe, BlazePose, YOLO-Pose

### 2.2 MediaPipe Pose (Top-Down)
- [ ] Architektur: BlazePose (Two-Step: Detection + Landmark)
- [ ] 33 Keypoints, inkl. Gesicht und Hände
- [ ] Pseudo-3D Output (z-Koordinate als relative Tiefe)
- [ ] Verfügbar für Mobile (TFLite) und Web

### 2.3 MoveNet (Bottom-Up)
- [ ] Architektur: Heatmap-basiert, Bottom-Up
- [ ] 17 COCO Keypoints
- [ ] Varianten: Lightning (schnell) vs. Thunder (genau)
- [ ] Optimiert für Echtzeit-Anwendungen

### 2.4 YOLOv8-Pose (One-Stage)
- [ ] Architektur: Anchor-free One-Stage Detection + Keypoints
- [ ] 17 COCO Keypoints
- [ ] Varianten: nano, small, medium, large, xlarge
- [ ] State-of-the-art Performance bei hoher Geschwindigkeit

### 2.5 Bestehende Benchmark-Studien
- [ ] COCO Keypoint Benchmark - Limitationen (hauptsächlich frontale Ansichten)
- [ ] Existierende Vergleichsstudien der Modelle
- [ ] **Forschungslücke identifizieren:** Systematische Rotation-Analyse fehlt

### 2.6 Viewpoint-Variation in Pose Estimation
- [ ] Bestehende Arbeiten zu Multi-View Pose Estimation
- [ ] Studien zu Okklusion und Self-Occlusion
- [ ] Gap: Quantitative Analyse der Winkelabhängigkeit bei 2D-Modellen

---

## 3. Methodik

### 3.1 Dataset

**REHAB24-6** (Zenodo)
- 65 Recordings, 184.825 Frames @ 30 FPS
- Ground Truth: 26 Skeleton Joints (3D Motion Capture → 2D projiziert)
- RGB Videos von 2 Kameras
- **Eignung:** Enthält natürliche Körperrotationen während Rehabilitationsübungen

**Begründung der Dataset-Wahl:**
- Ground Truth durch Motion Capture (Gold Standard)
- Reale Bewegungsszenarien (nicht gestellte Posen)
- Ausreichend große Datenmenge für statistische Signifikanz

### 3.2 Modellauswahl & Architekturvergleich

| Modell | Architektur | Ansatz | Keypoints |
|--------|-------------|--------|-----------|
| **MediaPipe Pose** | Two-Stage, Top-Down | Erst Person detecten, dann Keypoints | 33 |
| **MoveNet Thunder** | Heatmap-basiert, Bottom-Up | Alle Keypoints finden, dann gruppieren | 17 |
| **YOLOv8-Pose** | Anchor-free, One-Stage | Detection + Keypoints in einem Schritt | 17 |

**Begründung:**
- Drei fundamental verschiedene Architektur-Ansätze
- Alle drei sind smartphone-tauglich (On-Device Inference möglich)
- Repräsentieren aktuellen State-of-the-Art für Mobile Pose Estimation

### 3.3 Berechnung des Rotationswinkels

```
Rotationswinkel θ = arctan2(Δz, Δx) der Schulterachse

wobei:
- Δz = z_rechte_schulter - z_linke_schulter (Tiefe)
- Δx = x_rechte_schulter - x_linke_schulter (Breite)

θ = 0°  → Person steht frontal zur Kamera
θ = 90° → Person steht seitlich zur Kamera
```

**Wichtig:** Rotationswinkel wird aus **Ground Truth 3D-Daten** berechnet (nicht aus Predictions).

### 3.4 Keypoint Mapping

| Körperteil | Ground Truth | MediaPipe | MoveNet | YOLOv8 |
|------------|--------------|-----------|---------|--------|
| Nase | ? | 0 | 0 | 0 |
| Linkes Auge | ? | 2 | 1 | 1 |
| Rechtes Auge | ? | 5 | 2 | 2 |
| Linkes Ohr | ? | 7 | 3 | 3 |
| Rechtes Ohr | ? | 8 | 4 | 4 |
| Linke Schulter | ? | 11 | 5 | 5 |
| Rechte Schulter | ? | 12 | 6 | 6 |
| Linker Ellbogen | ? | 13 | 7 | 7 |
| Rechter Ellbogen | ? | 14 | 8 | 8 |
| Linkes Handgelenk | ? | 15 | 9 | 9 |
| Rechtes Handgelenk | ? | 16 | 10 | 10 |
| Linke Hüfte | ? | 23 | 11 | 11 |
| Rechte Hüfte | ? | 24 | 12 | 12 |
| Linkes Knie | ? | 25 | 13 | 13 |
| Rechtes Knie | ? | 26 | 14 | 14 |
| Linker Knöchel | ? | 27 | 15 | 15 |
| Rechter Knöchel | ? | 28 | 16 | 16 |

→ **Gemeinsame Schnittmenge:** 17 COCO Keypoints (MoveNet & YOLOv8 identisch, MediaPipe Subset)

### 3.5 Fehlermetrik

**Normalized Mean Per Joint Position Error (NMPJPE):**

```
NMPJPE = (1/N) * Σ ||gt_i - pred_i||₂ / torso_length
```

- Euklidische Distanz zwischen Ground Truth und Prediction
- Normalisiert durch Torso-Länge (Schulter-Hüfte) für Vergleichbarkeit
- Einheit: Prozent der Torso-Länge

**Weitere Metriken:**
- PCK@0.1 (Percentage of Correct Keypoints, Threshold = 10% Torso-Länge)
- Per-Joint Error (um anfällige Körperregionen zu identifizieren)

### 3.6 Experimentelles Design

**Unabhängige Variable:**
- Rotationswinkel θ (in 5°-Bins: 0-5°, 5-10°, ..., 85-90°)

**Abhängige Variablen:**
- NMPJPE
- PCK@0.1
- Per-Joint Error

**Kontrollierte Faktoren:**
- Gleiches Dataset für alle Modelle
- Gleiche Frames (synchron)
- Gleiche Keypoint-Auswahl (Schnittmenge)

**Analyse:**
1. Deskriptive Statistik pro Winkel-Bin
2. Visualisierung: Error vs. Rotationswinkel
3. Signifikanztests: Unterschied zwischen Modellen pro Bin (ANOVA + Post-hoc)
4. Regression: Modellierung des Fehleranstiegs

---

## 4. Implementierung

### 4.1 Pipeline-Architektur

```
┌─────────────────────────────────────────────────────────────────┐
│                        Video Frames                              │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   MediaPipe   │    │    MoveNet    │    │  YOLOv8-Pose  │
│  (Top-Down)   │    │  (Bottom-Up)  │    │  (One-Stage)  │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │   Keypoint    │
                    │    Mapper     │
                    └───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Unified Keypoints (17 COCO format)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────┐                         ┌───────────────┐
│  Ground Truth │                         │  Ground Truth │
│      2D       │                         │      3D       │
└───────────────┘                         └───────────────┘
        │                                           │
        ▼                                           ▼
┌───────────────┐                         ┌───────────────┐
│    Error      │                         │   Rotation    │
│  Calculation  │                         │    Angle      │
└───────────────┘                         └───────────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │  Aggregation  │
                    │  by Angle Bin │
                    └───────────────┘
```

### 4.2 Software-Architektur

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float
    name: str

class PoseEstimator(ABC):
    """Abstrakte Basisklasse für alle Pose Estimation Modelle."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Keypoint]:
        """Führt Inference auf einem Frame aus."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Gibt den Modellnamen zurück."""
        pass

    @property
    @abstractmethod
    def keypoint_mapping(self) -> dict[int, str]:
        """Mapping von Model-Index zu COCO Keypoint Name."""
        pass

class MediaPipeEstimator(PoseEstimator):
    """MediaPipe Pose Implementation."""
    ...

class MoveNetEstimator(PoseEstimator):
    """MoveNet Thunder Implementation."""
    ...

class YOLOPoseEstimator(PoseEstimator):
    """YOLOv8-Pose Implementation."""
    ...

class EvaluationPipeline:
    """Hauptklasse für die Evaluation."""

    def __init__(self, estimators: list[PoseEstimator]):
        self.estimators = estimators

    def run(self, video_path: str, gt_2d: np.ndarray, gt_3d: np.ndarray):
        """Führt Evaluation für alle Modelle durch."""
        ...
```

**Vorteile dieser Architektur:**
- Modelle sind austauschbar (weitere können einfach hinzugefügt werden)
- Parallele Ausführung möglich (jedes Modell in eigenem Thread/Prozess)
- Einheitliches Interface für Fehlerberechnung
- Testbar und wartbar

### 4.3 Technische Details

| Modell | Library | Variante | Input Size |
|--------|---------|----------|------------|
| MediaPipe | `mediapipe` | model_complexity=2 | variabel |
| MoveNet | TensorFlow Hub | Thunder | 256x256 |
| YOLOv8-Pose | `ultralytics` | yolov8m-pose | 640x640 |

**Koordinaten-Alignment:**
- Alle Modelle: Normalisierte Koordinaten [0,1] → Pixel-Koordinaten
- GT: Format zu verifizieren nach Dataset-Download

### 4.4 Datenverarbeitung

1. Frame-Extraktion aus Videos (OpenCV)
2. Parallele Inference mit allen Modellen
3. Keypoint-Mapping auf gemeinsame COCO-Schnittmenge
4. Rotationswinkel aus GT-3D berechnen
5. Fehlerberechnung pro Frame und Modell
6. Aggregation nach Winkel-Bins

---

## 5. Erwartete Ergebnisse

### 5.1 Hypothesen

**H1:** Der NMPJPE steigt mit zunehmendem Rotationswinkel bei allen drei Modellen.

**H2:** Der Fehleranstieg ist nicht-linear (stärker ab ~45°, wenn Self-Occlusion beginnt).

**H3:** Extremitäten (Handgelenke, Knöchel) zeigen höhere Fehler bei Rotation als Torso-Keypoints.

**H4:** Die drei Architekturansätze zeigen unterschiedliche Degradationsmuster:
- Top-Down (MediaPipe): Stabiler bei leichter Rotation, da explizite Person-Detection
- Bottom-Up (MoveNet): Anfälliger für Keypoint-Verwechslung bei Occlusion
- One-Stage (YOLO): Möglicherweise robuster durch End-to-End Training

**H5:** Es existiert ein kritischer Winkel θ_crit, ab dem alle Modelle signifikant degradieren.

### 5.2 Erwartete Visualisierungen

1. **Hauptgrafik:** Linienchart - NMPJPE vs. Rotationswinkel (alle 3 Modelle, mit Konfidenzintervallen)
2. **Heatmap:** Per-Joint Error über Rotationswinkel (pro Modell)
3. **Boxplots:** Fehlerverteilung pro Winkel-Bin
4. **Radar-Chart:** Modellvergleich über verschiedene Metriken
5. **Beispielbilder:** Qualitative Vergleiche bei 0°, 30°, 60°, 90°

---

## 6. Diskussion (Geplant)

### 6.1 Interpretation der Ergebnisse
- Bei welchem Winkel wird welches Modell unzuverlässig?
- Welche anatomischen/geometrischen Gründe erklären die Ergebnisse?
- Welcher Architekturansatz ist am robustesten gegenüber Rotation?

### 6.2 Praktische Implikationen
- **Für App-Entwickler:** Ab θ > X° sollte User-Feedback gegeben werden
- **Für Previa Health:** Konkrete Schwellenwerte für Bewegungsanalyse
- **Modellempfehlung:** Welches Modell für welchen Use-Case?

### 6.3 Limitationen
- Dataset aus Rehabilitationskontext (Generalisierbarkeit?)
- Nur Single-Person Szenarien
- Statische Winkelbetrachtung (keine dynamische Analyse)
- Indoor-Szenarien mit kontrollierter Beleuchtung

### 6.4 Ausblick
- Erweiterung auf weitere Modelle
- Dynamische Winkeländerungen während Bewegung
- Multi-Person Szenarien
- Outdoor / variable Beleuchtung

---

## 7. Zeitplan

| Phase | Aufgaben | Status |
|-------|----------|--------|
| **1. Grundlagen** | Dataset laden, Struktur verstehen, Keypoint-Mapping | ⬜ |
| **2. Pipeline** | Abstrakte Architektur + alle 3 Estimatoren implementieren | ⬜ |
| **3. Inference** | Pipeline über alle Videos laufen lassen | ⬜ |
| **4. Rotation** | Winkelberechnung aus GT, Frame-Annotation | ⬜ |
| **5. Evaluation** | Fehlerberechnung, Aggregation nach Bins | ⬜ |
| **6. Analyse** | Statistik, Visualisierungen, Hypothesentests | ⬜ |
| **7. Schreiben** | Thesis-Dokument verfassen | ⬜ |

---

## 8. Literatur (Vorläufig)

- Bazarevsky, V., et al. (2020). BlazePose: On-device Real-time Body Pose tracking.
- Google. (2021). MoveNet: Ultra fast and accurate pose detection model.
- Jocher, G., et al. (2023). Ultralytics YOLOv8.
- Lin, T. Y., et al. (2014). Microsoft COCO: Common Objects in Context.
- REHAB24-6 Dataset. Zenodo.
- [Weitere Literatur zu ergänzen]

---

## 9. Abhängigkeiten

```
python >= 3.10
mediapipe >= 0.10
tensorflow >= 2.13
tensorflow-hub
ultralytics >= 8.0
numpy
pandas
matplotlib
seaborn
scipy
opencv-python
```

---

## Notizen & Offene Punkte

- [ ] GT Joint-Namen aus `joints_names.txt` extrahieren
- [ ] Prüfen: Sind GT-Koordinaten in Pixel oder normalisiert?
- [ ] Entscheiden: Confidence-Threshold für Predictions?
- [ ] Klären: Beide Kameras separat oder zusammen auswerten?
- [ ] YOLOv8-Pose Variante festlegen (nano vs. small vs. medium)
