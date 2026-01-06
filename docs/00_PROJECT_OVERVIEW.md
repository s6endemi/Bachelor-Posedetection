# Project Overview: Pose Estimation Rotation Analysis

**Projekt:** Bachelorarbeit
**Autor:** Eren
**Start:** Januar 2026
**Status:** Pipeline validiert, bereit für Full-Run

---

## Forschungsfrage

> **Wie beeinflusst der Rotationswinkel einer Person relativ zur Kamera die Genauigkeit verschiedener 2D Pose Estimation Modelle?**

### Unterfragen
1. Ab welchem Rotationswinkel degradiert die Genauigkeit signifikant?
2. Gibt es systematische Unterschiede zwischen verschiedenen Modellarchitekturen?
3. Welche Körperregionen (Joints) sind besonders anfällig für rotationsbedingte Fehler?
4. Lassen sich praktische Schwellenwerte für mobile Anwendungen ableiten?

---

## Motivation

### Praktischer Kontext: Previa Health
- Startup für Bewegungsanalyse auf Smartphones
- Nutzer positionieren sich nicht immer optimal frontal zur Kamera
- **Kernproblem:** Es fehlt eine systematische Analyse, wie stark die Genauigkeit bei verschiedenen Rotationswinkeln degradiert
- **Ziel:** Konkrete Schwellenwerte für User-Feedback ("Bitte drehen Sie sich mehr zur Kamera")

### Wissenschaftliche Relevanz
- COCO Benchmark enthält hauptsächlich frontale Ansichten
- Systematische Rotation-Analyse fehlt in der Literatur
- Architekturvergleich unter kontrollierten Bedingungen

---

## Modelle im Vergleich

| Modell | Architektur | Ansatz | Variante |
|--------|-------------|--------|----------|
| **MediaPipe** | Top-Down | Erst Person detecten, dann Keypoints | Heavy (complexity=2) |
| **MoveNet** | Bottom-Up | Alle Keypoints finden, dann gruppieren | MultiPose Lightning |
| **YOLOv8-Pose** | One-Stage | Detection + Keypoints in einem Schritt | Nano/Medium |

### Warum diese drei?
1. **Drei fundamental verschiedene Architektur-Ansätze**
2. **Alle smartphone-tauglich** (On-Device Inference möglich)
3. **Repräsentieren aktuellen State-of-the-Art** für Mobile Pose Estimation
4. **Kostenlos und gut dokumentiert**

---

## Dataset: REHAB24-6

**Quelle:** Zenodo (öffentlich verfügbar)

### Statistiken
| Eigenschaft | Wert |
|-------------|------|
| Videos | 126 (2 Kameras × 63 Recordings) |
| Frames | ~185.000 @ 30 FPS |
| Probanden | Mehrere Personen bei Rehabilitationsübungen |
| Ground Truth | Motion Capture (26 Joints, 3D + 2D) |
| Kameras | c17 (horizontal), c18 (vertikal) |

### Warum REHAB24-6?
1. **Gold Standard Ground Truth:** Motion Capture System
2. **Natürliche Rotationen:** Rehabilitationsübungen beinhalten Drehungen
3. **Reale Szenarien:** Nicht gestellte Posen
4. **Multi-Person:** Manchmal Hintergrundpersonen (realistisch!)
5. **Ausreichend groß:** Statistische Signifikanz möglich

### Struktur
```
data/
├── videos/
│   ├── Ex1/
│   │   ├── PM_000-c17.avi    # Kamera 17, Proband 000
│   │   ├── PM_000-c18.avi    # Kamera 18, Proband 000
│   │   └── ...
│   ├── Ex2/
│   └── ...
└── ground_truth/
    ├── Ex1/
    │   ├── PM_000-c17.csv    # 2D Keypoints (Pixel)
    │   └── ...
    └── ...
```

---

## Code-Architektur

```
src/pose_evaluation/
├── estimators/
│   ├── base.py                        # Abstrakte Basisklasse
│   ├── mediapipe_estimator.py         # Torso-Size Selection
│   ├── movenet_multipose_estimator.py # BBox Selection
│   └── yolo_estimator.py              # BBox Selection
├── inference/
│   ├── data_loader.py                 # Findet Video-GT Paare
│   └── pipeline.py                    # Inference + Rotation
├── evaluation/
│   ├── metrics.py                     # NMPJPE, PCK
│   └── evaluator.py                   # Aggregation nach Winkel
└── data/
    └── keypoint_mapping.py            # 12 vergleichbare Joints
```

### Design-Prinzipien
1. **Abstrakte Basisklasse:** Alle Estimatoren implementieren gleiches Interface
2. **Modell-spezifische Selection:** Jedes Modell wählt Personen anders
3. **Trennung von Inference und Evaluation:** Predictions werden gespeichert
4. **Reproduzierbarkeit:** Alle Parameter dokumentiert

---

## Erwartete Ergebnisse

### Hypothesen
- **H1:** NMPJPE steigt mit zunehmendem Rotationswinkel
- **H2:** Fehleranstieg ist nicht-linear (stärker ab ~45°, Self-Occlusion)
- **H3:** Extremitäten zeigen höhere Fehler als Torso-Keypoints
- **H4:** Verschiedene Architekturen zeigen unterschiedliche Degradationsmuster
- **H5:** Es existiert ein kritischer Winkel θ_crit

### Geplante Visualisierungen
1. **Hauptgrafik:** NMPJPE vs. Rotationswinkel (3 Kurven + Konfidenzintervalle)
2. **Heatmap:** Per-Joint Error über Rotationswinkel
3. **Boxplots:** Fehlerverteilung pro Winkel-Bin
4. **Beispielbilder:** Qualitative Vergleiche bei 0°, 45°, 90°

---

## Abgrenzung

Diese Arbeit fokussiert sich auf:
- **2D Pose Estimation** (keine 3D-Rekonstruktion)
- **Single-Person Evaluation** (obwohl Multi-Person im Bild sein kann)
- **Statische Rotationswinkel** (Frame-für-Frame, keine Trajektorien)
- **On-Device Modelle** (keine Server-Modelle wie OpenPose)

---

## Relevanz für Previa Health

| Frage | Antwort durch diese Arbeit |
|-------|---------------------------|
| Welches Modell? | Vergleich unter realen Bedingungen |
| Ab wann User warnen? | Kritischer Winkel θ_crit |
| Welche Joints kritisch? | Per-Joint Analyse |
| Confidence-Threshold? | Empirisch getestet |

---

## Kontakt & Links

- **Repository:** `C:\Users\Eren\bachelor`
- **Dataset:** REHAB24-6 auf Zenodo
- **Dokumentation:** `docs/` Ordner
