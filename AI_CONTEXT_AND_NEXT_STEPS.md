# AI Context: Bachelor Thesis - Pose Estimation Evaluation

> **Zweck dieser Datei:** Vollstaendiger Kontext fuer AI-Sessions zur Fortsetzung der Bachelorarbeit.
> **Letzte Aktualisierung:** 11.01.2026 (Paper-Analyse: Baldinger et al. 2025 - Viewpoint Effects hinzugefuegt)
> **Aktueller Stand:** Evaluation abgeschlossen, **LITERATURVERGLEICH KOMPLETT** (6 Paper analysiert)

---

## 1. HINTERGRUND & MOTIVATION

### 1.1 Previa Health (Startup-Kontext)

**Was ist Previa Health?**
- Startup fuer smartphone-basierte Bewegungsanalyse
- Nutzer fuehren Physiotherapie-Uebungen zuhause durch
- Smartphone-Kamera analysiert die Bewegungen via Pose Estimation
- **Kernproblem:** Nutzer positionieren sich nicht immer optimal zur Kamera

**Praktische Relevanz:**
- Welches Modell ist am besten fuer mobile Anwendungen?
- Ab welchem Rotationswinkel sollte die App warnen?
- Wie robust sind die Modelle bei mehreren Personen im Bild?

### 1.2 Evolution der Forschungsfrage

**Urspruengliche Frage (zu simpel):**
> "Wie beeinflusst der Rotationswinkel die Genauigkeit von Pose Estimation?"

**Problem:** Diese Frage ist deskriptiv und die Hypothesen waren teilweise trivial.

**Was die Daten tatsaechlich zeigen:**
1. Rotationseffekt ist real aber moderat (+30-40% bei 60-90° vs 0-30°)
2. **Kein kritischer Winkel** - gradueller Anstieg, kein scharfer Knick
3. **Selection-Strategie ist wichtiger als Modellwahl**
4. **Multi-Person-Szenarien sind das Hauptproblem**, nicht Rotation

**Potentielle neue Richtungen:**
- Focus auf Selection-Robustheit bei Multi-Person
- Architekturvergleich: Top-Down vs Bottom-Up vs One-Stage
- Praktische Guidelines fuer mobile Reha-Apps

---

## 2. DATASET: REHAB24-6

**Quelle:** Zenodo (oeffentlich verfuegbar)
**Paper:** `docs/Rehab24-6.pdf`

| Eigenschaft | Wert |
|-------------|------|
| Videos | 126 (21 Patienten x 6 Uebungen x 2 Kameras) |
| Frames analysiert | 363.529 |
| Ground Truth | Motion Capture (optische Marker) |
| Kameras | c17 (frontal, 0-70°), c18 (lateral, 20-90°) |
| Besonderheit | 5 Videos mit Coach im Bild (Multi-Person) |

**Limitation:** Bimodale Rotationsverteilung - Personen stehen ENTWEDER frontal ODER seitlich, keine kontinuierliche Rotation waehrend der Uebungen.

---

## 3. EVALUIERTE MODELLE

| Modell | Architektur | Selection-Strategie | Bemerkung |
|--------|-------------|---------------------|-----------|
| **MediaPipe** | Top-Down | Torso-Groesse | Robust bei Multi-Person |
| **MoveNet** | Bottom-Up | BBox-Flaeche | Beste Genauigkeit |
| **YOLOv8-Pose** | One-Stage | BBox-Flaeche | Anfaellig bei Multi-Person |

Alle Modelle sind mobile-optimiert (On-Device Inference moeglich).

---

## 4. ZENTRALE ERGEBNISSE

### 4.1 Modell-Ranking (bereinigt, ohne >100% Frames)

| Modell | Mean NMPJPE | Median | Std |
|--------|-------------|--------|-----|
| **MoveNet** | **11.5%** | **10.4%** | 5.5% |
| MediaPipe | 12.5% | 11.2% | 7.2% |
| YOLO | 12.9% | 11.3% | 6.8% |

**Statistische Signifikanz:**
- MediaPipe vs MoveNet: **NICHT signifikant** (p=0.098)
- Beide signifikant besser als YOLO (p<0.001)
- Effect Sizes sind klein (Cohen's d < 0.1)

### 4.2 Selection-Robustheit (der wichtigste Fund!)

| Modell | Clean Mean | Coach Mean | Anstieg |
|--------|------------|------------|---------|
| **MediaPipe** | 14.5% | 44.8% | **+209%** |
| MoveNet | 14.8% | 64.9% | +340% |
| YOLO | 17.7% | 66.1% | +274% |

**Erkenntnis:** Torso-basierte Selection (MediaPipe) ist **~2x robuster** als BBox-Selection.

### 4.3 Rotationseffekt

| Bereich | Median NMPJPE | Interpretation |
|---------|---------------|----------------|
| 0-30° | ~9-11% | Optimal |
| 60-90° | ~13-15% | +30-40% Degradation |

**Wichtig:** Kein scharfer kritischer Winkel, gradueller Anstieg ab ~60°.

### 4.4 Kamera-Problem (c17 vs c18)

- c17 hat **10x mehr Person-Switch-Frames** (>100% NMPJPE)
- Nach Bereinigung: c17 (frontal) ~1-2% besser als c18 (lateral)
- Problem ist Multi-Person, nicht die Kamera selbst

---

## 5. RELEVANTE DATEIEN

### 5.1 Ergebnisse & Analysen
```
analysis/evaluation_results.tex      # LaTeX-Zusammenfassung (KORRIGIERT 11.01)
analysis/EVALUATION_SUMMARY.md       # Ausfuehrlicher Report
analysis/ANALYSIS_REPORT.md          # Detaillierter Report
analysis/results/frame_level_data.csv # 363k Frames, alle Metriken
analysis/results/rotation_analysis.csv # Rotation Buckets
```

### 5.2 Dokumentation
```
docs/00_PROJECT_OVERVIEW.md          # Projekt-Ueberblick
docs/01_METHODOLOGY.md               # Technische Details
docs/02_PROBLEMS_AND_SOLUTIONS.md    # Gefundene Probleme & Loesungen
```

### 5.3 Paper (fuer Literaturvergleich)
```
docs/Rehab24-6.pdf                   # REHAB24-6 Dataset Paper (WICHTIG!)
docs/s00530-021-00815-4.pdf          # HPE Paper
docs/s41598-025-29062-7.pdf          # Weiteres Paper
docs/sensors-23-08862.pdf            # Sensors Paper
docs/sensors-25-00799-v2.pdf         # Sensors Paper
docs/main.pdf                        # Thesis/Paper
```

### 5.4 Code
```
run_evaluation.py                    # Reproduzierbares Evaluation-Script
analysis/comprehensive_analysis.py   # Haupt-Analyse
src/pose_evaluation/                 # Pipeline-Code
```

---

## 6. BISHERIGE ERKENNTNISSE AUS LITERATUR

### 6.1 REHAB24-6 Paper (Cernek et al. 2025)

Das Original-Paper zum Dataset evaluiert:
- MediaPipe (Full)
- YOLOv8-Pose (verschiedene Groessen)
- RTMPose (NICHT von uns evaluiert)

**Was das Paper NICHT macht:**
- MoveNet wird NICHT getestet
- Selection-Strategien werden NICHT verglichen
- Multi-Person-Robustheit wird NICHT quantifiziert

**Unsere Contribution gegenueber dem Paper:**
1. MoveNet auf REHAB24-6 evaluiert (neu!)
2. Selection-Strategie-Vergleich (Torso vs BBox)
3. Multi-Person-Robustheit quantifiziert (+209% vs +340%)
4. Praktische Empfehlungen fuer mobile Reha-Apps

### 6.2 Roggio et al. (2024) - Narrative Review zu ML Pose Estimation

**Paper:** `docs/main.pdf`
**Titel:** "A comprehensive analysis of the machine learning pose estimation models used in human movement and posture analyses"
**Journal:** Heliyon 10 (2024) e39977
**Typ:** Narrative Review

#### Was das Paper abdeckt:
- Umfassender Ueberblick ueber alle gaengigen PEMs (OpenPose, PoseNet, AlphaPose, DeepLabCut, HRNet, MediaPipe, BlazePose, EfficientPose, MoveNet)
- Anwendungsbereiche: Posture Analysis, Gait Analysis, Sports, Injury Prevention, Remote Fitness Tracking, Ergonomics
- Modellvergleich nach Features und Anwendbarkeit

#### Relevante Modell-Informationen (Table 1 im Paper):

| Model | Landmarks | 2D/3D | Mobile | Pre-trained |
|-------|-----------|-------|--------|-------------|
| MediaPipe | 33 | **3D** | **Yes** | Internal |
| MoveNet | 17 | 2D | **Yes** | COCO |
| OpenPose | 135 | 2D | Limited | COCO, MPII |
| BlazePose | 33+ | 3D | Yes | Internal |

#### Anwendbarkeit laut Paper (Table 3):

| Model | Remote Fitness | Gait Analysis | Ergonomics |
|-------|----------------|---------------|------------|
| MediaPipe | **High** | High | Moderate |
| MoveNet | **High** | Medium | Moderate |
| OpenPose | Medium | High | High |

→ MediaPipe und MoveNet werden explizit fuer Remote Fitness/Rehabilitation empfohlen!

#### Accuracy gegen Gold Standard (aus dem Paper):

| Model | Metric | Wert | Quelle |
|-------|--------|------|--------|
| MediaPipe | Pearson r (lower limb) | 0.80 ± 0.1 | Lafayette et al. |
| MediaPipe | Pearson r (upper limb) | 0.91 ± 0.08 | Lafayette et al. |
| MoveNet | RMSE (knee angle) | 3.24° ± 1.19° | Bajpai & Joshi |
| OpenPose/AlphaPose | Systematic error | ~1-5mm | Needham et al. |

#### Kritische Zitate fuer unsere Thesis:

**Zu Multi-Person (S. 5):**
> "challenges like occlusion and crowded scenes hinder performance"

**Zu Limitationen (S. 6):**
> "the accuracy of landmarks detection is influenced by factors such as training of the ML model, source quality, or video occlusions"

**Zu Remote Rehabilitation (S. 7):**
> "ML PEMs for posture, gait, and movement analysis offer a promising approach for analyzing biomechanics during remote fitness or rehabilitation sessions"

**Zu MoveNet (S. 4):**
> "MoveNet [...] pushed the boundaries of real-time pose estimation with its exceptional speed and accuracy [...] employed in interactive fitness experiences"

#### GAP ANALYSIS - Was das Review NICHT macht (unsere Contribution!):

| Aspekt | Roggio et al. | Unsere Arbeit |
|--------|---------------|---------------|
| Multi-Person Quantifizierung | Erwaehnt als Problem, NICHT quantifiziert | ✓ +209% vs +340% Fehleranstieg |
| Selection-Strategien | NICHT verglichen | ✓ Torso vs BBox empirisch |
| Rotation-Analyse | NICHT erwaehnt | ✓ 0-90° systematisch |
| Echtes Reha-Dataset mit MoCap GT | Nur Literatur-Review | ✓ REHAB24-6, 363k Frames |
| MoveNet auf Reha-Daten | Keine empirischen Daten | ✓ Erstmals evaluiert |

#### Bedeutung fuer unsere Thesis:

1. **Legitimiert unsere Modellwahl:** Paper empfiehlt MediaPipe & MoveNet fuer Remote Fitness
2. **Identifiziert die Luecke:** Multi-Person wird als Problem genannt, aber nicht quantifiziert
3. **Unterstuetzt Previa Health Use Case:** Remote Rehabilitation als wichtiges Anwendungsfeld
4. **Wir liefern empirische Daten** zu dem, was das Review nur theoretisch beschreibt

---

### 6.3 Debnath et al. (2022) - Survey CV-basierte Rehabilitation

**Paper:** `docs/s00530-021-00815-4.pdf`
**Titel:** "A review of computer vision-based approaches for physical rehabilitation and assessment"
**Journal:** Multimedia Systems 28:209-239
**Typ:** Comprehensive Survey (164 Referenzen)
**Relevanz fuer uns:** ⭐⭐⭐ (Background, nicht zentral)

#### Was das Paper abdeckt:
- Umfassende Taxonomie von CV-basierter Rehabilitation (Virtual/Direct Rehab, Assessment)
- Historischer Ueberblick ueber 20 Jahre Forschung
- Fokus auf **Kinect-basierte** Systeme (Kinect ist deprecated!)
- Feature Encoding (DTW, HMM, kinematische Parameter)
- Datasets: SPHERE, UI-PRMD, KIMORE (alle relativ klein)

#### Wichtige Erkenntnisse:

**Zur Relevanz von Rehabilitation (S. 209):**
> "Statistics show that informal care for rehabilitation is the reason behind 27% of the whole treatment cost"

**Zum Dataset-Problem (S. 210):**
> "Due to difficulty in accessing patients, ethical issues [...] data is difficult to acquire and the datasets are often small"

**Zur Wichtigkeit von Joint Positions (S. 210):**
> "accurate body joint position estimation is vital for vision-based rehabilitation and assessment"

**Zu Future Work (S. 233):**
> "research in this domain needs publicly available large-scale datasets to take advantage of modern DL methods"

#### GAP - Was das Survey NICHT abdeckt (unsere Contribution!):

| Aspekt | Debnath et al. | Unsere Arbeit |
|--------|----------------|---------------|
| Mobile Modelle | NICHT erwaehnt | ✓ MediaPipe, MoveNet, YOLO |
| Person-Selection | NICHT diskutiert | ✓ Torso vs BBox verglichen |
| Rotationseffekte | NICHT systematisch | ✓ 0-90° in 10°-Bins |
| Modernes Dataset | Kinect-basiert | ✓ REHAB24-6 mit MoCap GT |

#### Bedeutung fuer unsere Thesis:
1. **Legitimiert das Forschungsfeld** - Rehabilitation ist wichtig und teuer
2. **Zeigt die technologische Luecke** - Kinect ist veraltet, mobile Modelle fehlen
3. **Bestaetigt Multi-Person als Problem** - aber nicht quantifiziert
4. **Nützlich für Related Work/Background** - nicht zentral für Contribution

---

### 6.4 Ullah et al. (2025) - Real-Time Action Scoring fuer Physiotherapie

**Paper:** `docs/s41598-025-29062-7.pdf`
**Titel:** "A real time action scoring system for movement analysis and feedback in physical therapy using human pose estimation"
**Journal:** Scientific Reports 15:44784 (2025) - **Sehr aktuell!**
**Typ:** Empirische Studie mit System-Entwicklung
**Relevanz fuer uns:** ⭐⭐⭐⭐ (Sehr relevant - gleiches Anwendungsfeld, gleiche Modelle)

#### Was das Paper abdeckt:
- Action Scoring System fuer Physiotherapie-Uebungen
- Verwendet **MediaPipe** fuer Pose Estimation (33 Keypoints)
- **Angular-based Analysis** statt direkter Keypoint-Positionen (robuster gegen Noise)
- **DTW (Dynamic Time Warping)** fuer temporale Ausrichtung
- **NCC (Normalized Cross-Correlation)** fuer Bewegungsvergleich
- Vergleich mit RepNet (Deep Learning Repetition Counter)
- Real-time Feedback fuer Home-based Rehabilitation

#### Wichtige Erkenntnisse & Zitate:

**Zur Home-based Rehabilitation (S. 1):**
> "Studies indicate that approximately **90% of rehabilitation exercises are performed at home** without direct supervision"

**Zur klinischen Genauigkeit (S. 1):**
> "Physiotherapists [...] achieve only **12° of accuracy** in angular measurements [...] with significantly lower accuracy for dynamic movements"

**Zur Markerless HPE Genauigkeit (S. 2):**
> "Liang et al. demonstrate that a 3D pose estimation system [...] produce joint angle measurements **within ±5° of a gold-standard** marker-based system"

**Zu Herausforderungen (S. 2):**
> "pose estimation accuracy is influenced by external factors such as **camera angle variations**, lighting conditions, and **subject-to-camera distance**"

**Zum Vorteil von Angular-based Analysis (S. 2):**
> "angular-based movement analysis [...] reduces sensitivity to occlusions and external disturbances, offering a more stable and consistent means of assessing movement quality"

**Zur Wahl von MediaPipe (S. 5):**
> "MediaPipe was selected [...] due to its **computational efficiency, real-time processing capability, and ease of deployment on resource-constrained devices**"

**Zu Limitationen (S. 17):**
> "the skeletal model provided by MediaPipe is inherently simplified. The system represents the human body with 33 keypoints, which do not capture the **full complexity of spinal articulation**"

#### Ergebnisse des Papers:

| Metrik | Wert | Bemerkung |
|--------|------|-----------|
| Keypoint Detection Accuracy | <10% Deviation | vs. manuell annotierte GT |
| ROM Deviation | 1.6% - 3.2% | vs. Kliniker-Messungen |
| Repetition Counting Error | ~7.5% | Durchschnitt ueber 4 Uebungen |
| DTW Alignment Accuracy | >90% | Temporale Ausrichtung |
| NCC Similarity | >0.85 | Bewegungsaehnlichkeit |

#### GAP - Was das Paper NICHT macht (unsere Contribution!):

| Aspekt | Ullah et al. (2025) | Unsere Arbeit |
|--------|---------------------|---------------|
| Modelle verglichen | **NUR MediaPipe** | ✓ MediaPipe, MoveNet, YOLO |
| Dataset | 6 Probanden, kontrollierte Demo | ✓ 126 Videos, echte Reha-Patienten |
| Ground Truth | Kliniker visuell | ✓ Motion Capture 3D |
| Rotationsanalyse | NICHT systematisch | ✓ 0-90° in 10°-Bins |
| Multi-Person | NICHT adressiert | ✓ Selection-Strategien verglichen |
| Selection-Robustheit | NICHT untersucht | ✓ Torso vs BBox quantifiziert |
| Viewpoint-Effekte | "camera angle" nur erwaehnt | ✓ c17 vs c18 empirisch analysiert |

#### Bedeutung fuer unsere Thesis:

1. **Validiert MediaPipe fuer Physiotherapie** - Gleiches Modell wie wir, bestätigt Eignung
2. **Zeigt die Luecke** - Kein Modellvergleich, keine Rotationsanalyse
3. **Bestaetigt Angular-based als robust** - Unsere NMPJPE nutzt auch normierte Metriken
4. **Kontrollierte vs. echte Daten** - Ihr Dataset ist klein und kontrolliert, unseres ist realistischer
5. **Sehr aktuell (2025)** - Zeigt, dass das Thema hochrelevant ist
6. **Komplementaere Arbeit** - Sie fokussieren auf Scoring/Feedback, wir auf Modell-Accuracy

#### Wichtige Referenzen aus dem Paper fuer uns:

- **Liang et al. (2022)** - 3D Pose Estimation Genauigkeit im Gait Analysis
- **Abbott et al. (2022)** - Kliniker-Genauigkeit bei Winkelmessung (12°)
- **Aguilar-Ortega et al. (2023)** - UCO Physical Rehabilitation Dataset (sensors-23-08862.pdf!)
- **Roggio et al. (2024)** - ML PEM Review (bereits analysiert)

---

### 6.5 Aguilar-Ortega et al. (2023) - UCO Physical Rehabilitation Dataset ⭐⭐⭐⭐⭐

**Paper:** `docs/sensors-23-08862.pdf`
**Titel:** "UCO Physical Rehabilitation: A Pose Estimation Dataset"
**Journal:** Sensors 2023, 23, 8862
**Typ:** Dataset Paper + Multi-Model Benchmark
**Relevanz fuer uns:** ⭐⭐⭐⭐⭐ (HOECHSTE RELEVANZ - direktester Vergleich!)

#### Warum dieses Paper so wichtig ist:

Dies ist das **nächstvergleichbare Paper** zu unserer Arbeit. Es macht fast genau das, was wir tun: Multiple HPE-Modelle auf einem Rehabilitation-Dataset mit Motion Capture Ground Truth evaluieren. Die Luecken, die es laesst, sind genau unsere Contribution!

---

#### Das UCO Dataset im Detail:

| Eigenschaft | UCO Dataset | REHAB24-6 (unser Dataset) |
|-------------|-------------|---------------------------|
| **Probanden** | 27 (17 Training, 10 Test) | 21 Patienten |
| **Videos** | 2160 | 126 |
| **Frames** | ~108.000 | 363.529 (3x mehr!) |
| **Kameras** | 5 Viewpoints | 2 Kameras (c17, c18) |
| **Uebungen** | 16 Physio-Uebungen | 6 Uebungen |
| **Ground Truth** | OptiTrack IR (±0.5mm) | Motion Capture (optisch) |
| **Probanden-Typ** | **Gesunde Freiwillige** | **Echte Reha-Patienten** |
| **Verfuegbarkeit** | Oeffentlich (Zenodo) | Oeffentlich (Zenodo) |

**KRITISCHER UNTERSCHIED:** UCO nutzt gesunde Probanden, REHAB24-6 echte Patienten!

---

#### Koennten wir das UCO Dataset auch nutzen?

**JA - Potentieller Mehrwert:**

| Aspekt | Vorteil | Aufwand |
|--------|---------|---------|
| 5 Kamera-Winkel | Mehr Rotationsvariation | Mittlerer Aufwand (5x mehr Videos) |
| 16 Uebungen | Breitere Bewegungspalette | Hoher Aufwand |
| Cross-Dataset Validierung | Staerkere Generalisierung | Geringer Aufwand |
| Mehr Probanden | Bessere Statistik | - |

**ABER - Limitationen:**

1. **Gesunde Probanden:** Weniger realistisch fuer echte Reha-Szenarien
2. **Keine Multi-Person Szenarien:** Unser wichtigster Fund (Selection-Robustheit) waere nicht testbar
3. **Zeitaufwand:** Komplette Pipeline auf neuem Dataset ausfuehren = mehrere Tage
4. **Scope Creep:** Bachelor-Thesis sollte fokussiert bleiben

**EMPFEHLUNG:** Als Future Work erwaehnen, aber NICHT fuer diese Thesis implementieren.

---

#### Getestete Modelle im Paper (8 Stueck):

| Modell | Architektur | 2D/3D | Unser Vergleich |
|--------|-------------|-------|-----------------|
| **AlphaPose** | Top-Down | 2D | Nicht evaluiert |
| **MediaPipe** | Top-Down | 3D | ✓ Evaluiert |
| **HMR** | End-to-End | 3D | Nicht evaluiert |
| **VideoPose3D** | Lifting | 3D | Nicht evaluiert |
| **KAPAO** | One-Stage | 2D | Nicht evaluiert |
| **HybrIK** | Top-Down | 3D | Nicht evaluiert |
| **StridedTransformer** | Transformer | 3D | Nicht evaluiert |
| **PoseBERT** | Transformer | 3D | Nicht evaluiert |

**KRITISCHE LUECKE - NICHT GETESTET:**
- ❌ **MoveNet** - Wir evaluieren es erstmals auf Reha-Daten!
- ❌ **YOLOv8-Pose** - Wir evaluieren es!
- ❌ **Mobile-optimierte Modelle** - Fokus auf Server-Modelle

---

#### Wichtige Ergebnisse aus dem Paper:

**Kamera-Viewpoint-Effekt (Tabelle 5):**

| Kamera | Mean Error (mm) | Interpretation |
|--------|-----------------|----------------|
| Frontal (0°) | ~45mm | Beste Genauigkeit |
| Semi-lateral (45°) | ~55mm | +22% Degradation |
| Lateral (90°) | ~80mm | +78% Degradation |

→ **Bestaetigt unsere Rotation-Findings!** Seitliche Ansichten sind schlechter.

**Per-Joint Analyse:**
- Handgelenke und Ellenbogen haben hoechste Fehler
- Hueften und Schultern sind stabiler
→ **Deckt sich mit unseren Ergebnissen!**

---

#### Kritische Zitate fuer unsere Thesis:

**Zur Viewpoint-Problematik (S. 2):**
> "Regarding the camera positioning, the field of view is one the most important factors that affects pose estimation accuracy"

**Zur Notwendigkeit von Multi-View (S. 3):**
> "Multi-camera systems provide higher accuracy because each camera can capture different body parts"

**Zu Occlusionen (S. 14):**
> "The results also show that full-body occlusions strongly deteriorate the performance of all models"

**Zur praktischen Relevanz (S. 15):**
> "Physical rehabilitation exercises performed at home [...] require robust pose estimation that can handle sub-optimal camera positions"

---

#### GAP ANALYSIS - Was das Paper NICHT macht (unsere Contribution!):

| Aspekt | UCO Paper (2023) | Unsere Arbeit |
|--------|------------------|---------------|
| **MoveNet getestet** | ❌ NEIN | ✓ Erstmals auf Reha-Daten! |
| **YOLO getestet** | ❌ NEIN | ✓ YOLOv8-Pose Nano |
| **Mobile-Fokus** | ❌ Server-Modelle | ✓ On-Device Modelle |
| **Multi-Person** | ❌ Single-Person | ✓ Coach-Szenarien analysiert |
| **Selection-Strategie** | ❌ Nicht relevant (1 Person) | ✓ Torso vs BBox verglichen |
| **Echte Patienten** | ❌ Gesunde Probanden | ✓ Reha-Patienten |
| **Selection-Robustheit quantifiziert** | ❌ - | ✓ +209% vs +340% |

---

#### Bedeutung fuer unsere Thesis:

1. **Staerkste Validierung unserer Arbeit:**
   - Paper zeigt Camera-Viewpoint-Effekt, wir quantifizieren ihn fuer mobile Modelle
   - Paper testet NICHT MoveNet/YOLO - wir schliessen diese Luecke

2. **Positionierung:**
   - UCO = Server-Modelle, gesunde Probanden, Single-Person
   - Unsere Arbeit = Mobile-Modelle, echte Patienten, Multi-Person

3. **Zitierbar fuer:**
   - "Viewpoint significantly affects HPE accuracy" (validiert)
   - "Full-body occlusions deteriorate performance" (bestaetigt unsere Multi-Person Findings)

4. **Unsere Unique Contribution gegenueber UCO:**
   - Erste MoveNet-Evaluation auf Rehabilitation-Daten
   - Selection-Robustheit quantifiziert (fehlt komplett in UCO)
   - Echte Patienten statt gesunde Freiwillige
   - Mobile-optimierte Modelle fuer Smartphone-Anwendungen

---

#### Cross-Reference zu anderen Papers:

- **Ullah et al. (2025)** zitiert dieses Paper als Related Work
- **Roggio et al. (2024)** listet einige der UCO-Modelle (HMR, AlphaPose)
- **REHAB24-6 Paper** ist neuer und testet teilweise andere Modelle

---

### 6.6 Baldinger et al. (2025) - Camera Viewing Angle Effects on OpenPose ⭐⭐⭐⭐

**Paper:** `docs/sensors-25-00799-v2.pdf`
**Titel:** "Influence of the Camera Viewing Angle on OpenPose Validity in Motion Analysis"
**Journal:** Sensors 2025, 25, 799 (Januar 2025 - **SEHR AKTUELL!**)
**Institution:** Technical University of Munich
**Typ:** Empirische Validierungsstudie
**Relevanz fuer uns:** ⭐⭐⭐⭐ (Sehr relevant - validiert Viewpoint-Effekte direkt!)

#### Warum dieses Paper wichtig ist:

Dies ist die **aktuellste Studie** (Januar 2025) zum Thema Kamerawinkel und HPE-Genauigkeit. Sie untersucht systematisch, wie verschiedene Kamerawinkel die Gelenkwinkel-Schaetzung beeinflussen - genau das, was wir fuer Previa Health brauchen!

---

#### Studiendesign im Detail:

| Aspekt | Details |
|--------|---------|
| **Probanden** | 20 gesunde Teilnehmer (16 nach Ausschluss) |
| **Modell** | **NUR OpenPose** (25 Keypoints) |
| **Ground Truth** | Vicon Marker-basiert MoCap (200 Hz, 12 Kameras) |
| **RGB Kameras** | 4 iPad Pro (43 fps, 1194x834 px) |
| **Kamera-Setup** | 4 Winkel bei 45°-Intervallen um den Probanden |
| **Uebung** | Front Lunges (6 Wiederholungen) |
| **Gelenke analysiert** | Knie, Huefte, Ellenbogen, Schulter (Flexion/Extension) |

**Kamera-Positionen:**
```
                    90° (Bewegungsrichtung)
                         ↑
    Back Left (135°)     |     Front Left (45°)
                    \    |    /
                     \   |   /
        180° ←--------[Person]--------→ 0°
                     /   |   \
                    /    |    \
    Back Right (225°)    |     Front Right (315°)
                         ↓
                       270°
```

---

#### Kern-Ergebnisse des Papers:

**1. Korrelations-Analyse (ICC):**

| Gelenk | Front Right | Front Left | Back Left | Back Right |
|--------|-------------|------------|-----------|------------|
| **Knie** | 0.95 (exc) | 0.86 (good) | 0.95 (exc) | 0.96 (exc) |
| **Huefte** | 0.88 (good) | 0.74 (mod) | 0.94 (exc) | 0.94 (exc) |
| **Ellenbogen** | 0.75 (mod) | 0.80 (good) | 0.83 (good) | 0.84 (good) |
| **Schulter** | 0.28 (poor) | 0.72 (mod) | 0.59 (mod) | 0.55 (mod) |

→ **Back-viewing cameras zeigen beste ICCs!**
→ **Schulter ist problematisch aus allen Winkeln**

**2. RMSE (Root Mean Square Error in Grad):**

| Gelenk | Front Right | Front Left | Back Left | Back Right |
|--------|-------------|------------|-----------|------------|
| Knie | 18.2° | 29.4° | 16.7° | 15.5° |
| Huefte | 19.0° | 25.3° | 15.7° | 15.2° |
| Ellenbogen | 37.1° | 35.0° | 29.4° | 30.4° |
| Schulter | 36.5° | 23.6° | 28.3° | 28.5° |
| **Overall** | 31.8° | 29.9° | **23.4°** | **22.2°** |

→ **Back Right hat niedrigsten Overall RMSE (22.2°)**

**3. Prozentuale Abweichung bei Peak-Winkeln (Tabelle 6 im Paper):**

| Gelenk | Front Right | Front Left | Back Left | Back Right | Mean |
|--------|-------------|------------|-----------|------------|------|
| Knie | 20% | **28%** | 9% | 11% | 17% |
| Huefte | 18% | **30%** | 15% | 13% | 19% |
| Ellenbogen | 8% | 14% | 16% | 17% | 14% |
| Schulter | 21% | **38%** | 27% | 30% | 29% |
| **Mean** | 17% | **27%** | 17% | 18% | - |
| **Mean (ohne Schulter)** | 15% | 24% | **13%** | **13%** | - |

→ **Front Left ist SCHLECHTESTE Perspektive (27% Mean Deviation)**
→ **Back-viewing cameras: nur 13% Abweichung (ohne Schulter)**

---

#### Kritische Zitate fuer unsere Thesis:

**Zum Kamerawinkel-Effekt (Abstract):**
> "the analysis also revealed significant biases when comparing the joint angles inferred from the different viewing angles"

**Zur Back-View Empfehlung (S. 15):**
> "the back-viewing cameras demonstrate lower deviations, indicating that the back views are most suitable for this type of exercise estimation"

**Zur praktischen Anwendung (S. 3):**
> "Patients or trainees at home might lack an understanding of which angle yields the most reliable results"

**Zur Notwendigkeit von Viewpoint-Forschung (S. 3):**
> "The impact of alternative camera angles has not been thoroughly investigated and quantified, therefore limiting the applicability in diverse setups"

**Zum Home Training Kontext (S. 16):**
> "Using OpenPose in biomechanics research can help overcome some of the drawbacks of conventional marker-based MoCap systems, especially in terms of time and costs, making it a valuable tool, especially in home training or clinical environments"

**Zur Occlusion-Problematik (S. 14):**
> "Especially the arms are prone to occlusions by the torso, which makes an accurate estimation of the elbow and shoulder joints challenging"

---

#### Vergleich mit unseren Ergebnissen:

| Aspekt | Baldinger et al. (2025) | Unsere Arbeit |
|--------|-------------------------|---------------|
| **Viewpoint-Effekt** | 13-27% je nach Winkel | +30-40% bei 60-90° Rotation |
| **Beste Perspektive** | Back-viewing (13%) | Frontal (c17 bereinigt ~1-2% besser) |
| **Schlechteste Gelenke** | Schulter, Ellenbogen | Handgelenke, Hueften |
| **Schlussfolgerung** | Kamerawinkel wichtig | Rotation wichtig, aber Selection wichtiger |

**Wichtige Uebereinstimmungen:**
- Beide zeigen: **Kamerawinkel beeinflusst HPE-Genauigkeit signifikant**
- Beide zeigen: **Occlusions sind Hauptproblem**
- Beide zeigen: **Obere Extremitaeten (Schulter, Ellenbogen) sind problematischer**

**Wichtige Unterschiede:**
- Sie messen **diskrete Winkel** (45° Intervalle), wir messen **kontinuierliche Rotation** (0-90°)
- Sie analysieren **Gelenkwinkel** (Angular), wir analysieren **Keypoint-Positionen** (NMPJPE)
- Ihre Metrik ist RMSE in Grad, unsere ist NMPJPE in Prozent der Torso-Laenge

---

#### GAP ANALYSIS - Was das Paper NICHT macht (unsere Contribution!):

| Aspekt | Baldinger et al. (2025) | Unsere Arbeit |
|--------|-------------------------|---------------|
| **Modelle getestet** | ❌ NUR OpenPose | ✓ MediaPipe, MoveNet, YOLO |
| **MoveNet** | ❌ NICHT getestet | ✓ Erstmals auf Reha-Daten! |
| **Multi-Person** | ❌ Single-Person | ✓ Coach-Szenarien analysiert |
| **Selection-Strategie** | ❌ Nicht relevant | ✓ Torso vs BBox verglichen |
| **Dataset** | ❌ 16 gesunde, 1 Uebung | ✓ 21 Patienten, 6 Uebungen |
| **Frames** | ❌ ~1000 (geschaetzt) | ✓ 363.529 Frames |
| **Kontinuierliche Rotation** | ❌ Nur 4 diskrete Winkel | ✓ 0-90° in 10°-Bins |
| **Rehabilitation-Kontext** | ❌ Lunges (Fitness) | ✓ Echte Physio-Uebungen |
| **Mobile-Fokus** | ❌ OpenPose (Server) | ✓ On-Device Modelle |

---

#### Bedeutung fuer unsere Thesis:

1. **Staerkste externe Validierung unserer Viewpoint-Findings:**
   - Paper bestaetigt: Kamerawinkel hat signifikanten Einfluss auf HPE
   - Quantifiziert den Effekt (13-27% Abweichung je nach Winkel)
   - Sehr aktuell (Januar 2025) - zeigt Relevanz des Themas

2. **Positionierung unserer Arbeit:**
   - Baldinger = OpenPose, gesunde Probanden, Fitness-Uebung, 4 Winkel
   - Wir = Mobile Modelle, echte Patienten, Reha-Uebungen, kontinuierliche Rotation

3. **Zitierbar fuer:**
   - "Camera viewing angle significantly affects HPE accuracy" (S. 12)
   - "Back-viewing cameras performed best" (S. 15)
   - "Shoulder joint estimation is problematic from all angles" (S. 7)

4. **Unsere Unique Contribution gegenueber Baldinger:**
   - **Modellvergleich:** Wir testen 3 mobile Modelle, sie nur OpenPose
   - **Multi-Person:** Wir quantifizieren Selection-Robustheit (+209% vs +340%)
   - **Echte Reha-Daten:** 363k Frames von echten Patienten
   - **Praktische Guidelines:** Empfehlungen fuer Previa Health

5. **Komplementaere Arbeit:**
   - Sie: Detaillierte Gelenkwinkel-Analyse aus verschiedenen Perspektiven
   - Wir: Modellvergleich mit Fokus auf Multi-Person und Selection

---

#### Praktische Empfehlungen aus dem Paper (fuer Previa Health relevant):

1. **Back-viewing cameras bevorzugen** wenn frontal/sagittal nicht moeglich
2. **Schulter-Tracking vermeiden** oder mit Vorsicht interpretieren
3. **Occlusions minimieren** durch optimale Kamera-Positionierung
4. **Bewegungs-spezifische Kamerawinkel testen** vor dem Deployment

---

## 7. OFFENE FRAGEN FUER LITERATURVERGLEICH

### 7.1 Zu klaerende Fragen

1. **Was sagen andere Paper zu Rotation bei HPE?**
   - Gibt es Studien zur Rotations-Sensitivitaet?
   - Wurden kritische Winkel definiert?

2. **Selection-Strategien in der Literatur?**
   - Wie loesen andere das Multi-Person-Problem?
   - Gibt es Vergleiche von Torso vs BBox Selection?

3. **Mobile HPE fuer Rehabilitation?**
   - Welche Modelle werden in Reha-Apps verwendet?
   - Was sind akzeptable Fehlerraten fuer klinische Anwendungen?

4. **Architekturvergleiche?**
   - Top-Down vs Bottom-Up vs One-Stage unter welchen Bedingungen besser?
   - Robustheit bei Occlusion/Rotation?

### 7.2 Ziel des Literaturvergleichs

Finde heraus:
- Wo positioniert sich unsere Arbeit in der Literatur?
- Was ist unsere einzigartige Contribution?
- Welche Richtung sollte die Thesis einschlagen?

---

## 8. ANWEISUNGEN FUER DIE NAECHSTE AI-SESSION

### 8.1 Deine Rolle

Du bist ein **kritischer, ehrlicher Bachelor-Coach** ohne Sugarcoating. Du analysierst eigenstaendig und hilfst, eine qualitativ hochwertige Bachelorarbeit zu schreiben.

### 8.2 Aktueller Stand

- Evaluation ist **ABGESCHLOSSEN** (363k Frames, 3 Modelle, alle Metriken)
- LaTeX-Zusammenfassung ist **KORRIGIERT** (Tabelle 16 war falsch)
- Naechster Schritt: **LITERATURVERGLEICH**

### 8.3 Konkrete Aufgabe

1. **Lies die Paper** in `docs/` (PDF-Dateien)
2. **Vergleiche** mit unseren Ergebnissen
3. **Identifiziere** unsere einzigartige Contribution
4. **Empfehle** eine klare Thesis-Richtung

### 8.4 Wichtige Constraints

- Die Rotation-Analyse allein ist **NICHT ausreichend** fuer eine gute Thesis
- Der Selection-Robustheit-Fund ist **potentiell wertvoller**
- Das Dataset hat **Limitationen** (bimodale Rotation, kontrollierte Lab-Bedingungen)
- Previa Health braucht **praktische Empfehlungen**, nicht nur akademische Ergebnisse

### 8.5 Erwartetes Output

Nach dem Literaturvergleich:
1. Klare Positionierung der Arbeit
2. Definierte Contribution (was ist NEU?)
3. Konkrete Thesis-Struktur/Outline
4. Related Work Section Entwurf

---

## 9. QUICK REFERENCE

### Kern-Zahlen zum Merken

| Metrik | Wert |
|--------|------|
| Beste Genauigkeit | MoveNet 10.4% Median |
| Selection-Robustheit | MediaPipe 2x besser |
| Rotationseffekt | +30-40% bei 60-90° |
| Person-Switch-Problem | c17 hat 10x mehr als c18 |

### Empfehlungen fuer Previa Health

| Szenario | Empfehlung |
|----------|------------|
| Single-Person garantiert | MoveNet |
| Multi-Person moeglich | MediaPipe |
| Seitliche Ansicht (>60°) | Warnung ausgeben |

---

## 10. CHANGELOG

| Datum | Aenderung |
|-------|-----------|
| 11.01.2026 | **Baldinger et al. (2025) analysiert** - Camera Viewing Angle Effects (OpenPose, Viewpoint-Validierung) |
| 11.01.2026 | **Aguilar-Ortega et al. (2023) analysiert** - UCO Dataset (WICHTIGSTES Paper, direktester Vergleich!) |
| 11.01.2026 | **Ullah et al. (2025) analysiert** - Real-Time Action Scoring (MediaPipe, DTW/NCC) |
| 11.01.2026 | **Debnath et al. (2022) analysiert** - CV-based Rehabilitation Survey |
| 11.01.2026 | **Roggio et al. (2024) analysiert** - Narrative Review zu ML PEMs |
| 11.01.2026 | Kontextdatei erstellt |
| 11.01.2026 | LaTeX Tabelle 16 korrigiert (MediaPipe: 0-10°, nicht 10-20°) |
| 11.01.2026 | Umfassende Evaluation abgeschlossen |
| 09.01.2026 | Video-Kategorisierung (121 Clean, 5 Coach) |

---

## 11. NOCH ZU LESENDE PAPER

| Paper | Datei | Status | Relevanz |
|-------|-------|--------|----------|
| REHAB24-6 Dataset | `docs/Rehab24-6.pdf` | Teilweise analysiert | ⭐⭐⭐⭐⭐ |
| Roggio et al. (2024) | `docs/main.pdf` | ✓ Analysiert | ⭐⭐⭐⭐⭐ |
| Debnath et al. (2022) | `docs/s00530-021-00815-4.pdf` | ✓ Analysiert | ⭐⭐⭐ (Background) |
| Ullah et al. (2025) | `docs/s41598-025-29062-7.pdf` | ✓ Analysiert | ⭐⭐⭐⭐ (Sehr relevant) |
| Aguilar-Ortega et al. (2023) | `docs/sensors-23-08862.pdf` | ✓ Analysiert | ⭐⭐⭐⭐⭐ (Direktester Vergleich!) |
| Baldinger et al. (2025) | `docs/sensors-25-00799-v2.pdf` | ✓ Analysiert | ⭐⭐⭐⭐ (Viewpoint-Validierung) |

---

**NAECHSTER SCHRITT:** Literaturvergleich ist KOMPLETT! Nun: Thesis-Struktur definieren und Related Work Section schreiben.
