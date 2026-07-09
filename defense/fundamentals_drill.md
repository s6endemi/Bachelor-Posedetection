# Fundamentals Drill — CV/HPE-Wissen für die Gall-Fragerunde

> Wissen, das ÜBER den Thesis-Text hinausgeht. Regel: jede Modell-Frage eine Ebene
> tiefer beantworten können, als die Arbeit es verlangt. Alles auf Englisch drillen.

## 1. BlazePose / MediaPipe Pose (Bazarevsky et al. 2020)

**Pipeline: Detector → Tracker (two-stage, top-down).**
- Der Person-Detector ist laut BlazePose-Paper von **BlazeFace abgeleitet und
  face-anchored**: er detektiert primär das Gesicht als stärkstes Signal für die Person
  und sagt zusätzlich Alignment-Parameter voraus (u. a. Mid-Hip-Punkt,
  Skalierung/Rotation), aus denen die Person-ROI konstruiert wird.
  ⚠️ **KORREKTUR (7.7.2026, gegen finale Arbeit geprüft):** Die finale Thesis claimt
  KEINEN Face-Filter-Mechanismus. Kap. 6.3 lässt den Coach-Robustheits-Mechanismus
  explizit offen ("candidate retention, thresholding, or post-detection selection
  behavior"; "cannot be localized further"). Das Face-Detector-Wissen ist reines
  Q&A-Hintergrundwissen: als *konsistente, ungetestete* Hypothese kennzeichnen und
  betonen, dass die Thesis sie bewusst nicht claimt → Zwei-Ebenen-Antwort in
  qa_catalog.md B7.
- Der Pose-Estimator läuft auf der ausgerichteten ROI und liefert **33 Landmarks**
  (COCO-17 + Gesicht, Hände, Füße) mit visibility/presence-Werten.
- **Training kombiniert Heatmap- und Regression-Supervision; zur Inferenz läuft nur der
  Regression-Pfad** (Heatmap-Zweig wird abgeworfen) → schnell + sub-pixel, aber
  unimodal.
- Im Video-Modus ersetzt Tracking (ROI aus vorherigem Frame) den Detector; **in der
  Thesis: RunningMode.IMAGE → Detector läuft jeden Frame, kein temporaler Zustand.**
- Varianten Lite/Full/Heavy = unterschiedliche Backbone-Kapazität, gleiche TFLite-Runtime.

**Gall-Fragen dazu:**
- *"Why does MediaPipe only ever output one person?"* → Die klassische
  MediaPipe-Pose-Lösung ist eine Single-Person-Pipeline: ein Detektor wählt die
  dominante Person, der Estimator verarbeitet genau eine ROI (num_poses konfigurierbar
  in der neuen Tasks-API, aber Standard-Setup ist 1 — in der Thesis dokumentiert).
- *"What do the visibility scores mean?"* → Gelernte Wahrscheinlichkeit, dass der
  Landmark im Bild sichtbar/präsent ist — kein kalibriertes Konfidenzmaß.

## 2. MoveNet (Google 2021, TF-Blog/TF-Hub)

**Architektur: MobileNetV2-Backbone + Feature Pyramid Network, CenterNet-artiger Head.**
Vier Prediction-Heads:
1. **Person-Center-Heatmap** — wo sind Personenzentren?
2. **Keypoint-Regression-Field** — initiale 17 Keypoint-Positionen, regrediert vom Center
3. **Keypoint-Heatmaps** — full-image Heatmaps pro Keypoint-Typ
4. **Local-Offset-Field** — Sub-Pixel-Korrektur der Heatmap-Maxima

**Inferenz:** Center wählen → Keypoints regressieren → Refinement: Heatmap-Maxima,
gewichtet nach Distanz zur Regression (verhindert Springen auf fremde Personen) →
Offsets addieren. Ein einziger Forward-Pass, keine Crops → **kosten-invariant zur
Personenzahl = operationales bottom-up-Kriterium.**

- **Lightning** (Input 192×192, schneller) vs. **Thunder** (256×256, genauer) —
  SinglePose. **MultiPose Lightning:** bis zu 6 Personen + Bounding Boxes (in der
  Thesis die Haupt-Konfiguration).
- **Trainingsdaten laut Google: COCO + internes "Active"-Dataset (Yoga-, Fitness-,
  Tanz-Videos).** ⚠️ Starkes Diskussionsargument: MoveNet ist auf fitness-artige
  Bewegungen trainiert — plausibler Mit-Grund für seinen Reha-Vorsprung. Als Hypothese
  formulieren (Trainingsdaten nicht öffentlich verifizierbar).

## 3. YOLOv8-Pose (Jocher et al. 2023, Ultralytics)

- **One-stage, anchor-free** Objektdetektor mit zusätzlichem Pose-Head: pro Detection
  Box + 17 Keypoints (x, y, confidence) in einem Forward-Pass.
- Backbone mit **C2f-Blöcken** (CSP-inspiriert), **decoupled head** (Klassifikation und
  Regression getrennt), Task-Aligned Assigner für das Label-Assignment.
- Keypoint-Loss ist **OKS-basiert** (keypoint-spezifische Toleranzen), Box-Regression
  mit Distribution Focal Loss.
- Varianten n/s/m/l/x (Nano in der Thesis Haupt-Konfiguration); trainiert auf COCO-Pose.
- Runtime-Unterschied zur Konkurrenz: **PyTorch (Ultralytics), nicht TFLite.**

## 4. Heatmaps vs. direkte Regression (Kernkonzept — sicher können!)

| | Heatmap-basiert | Direkte Regression |
|---|---|---|
| Output | 2D-Wahrscheinlichkeitskarte pro Keypoint, argmax | Koordinaten direkt |
| Stärken | räumliche Unsicherheit explizit, multimodal (Ambiguität abbildbar), leichter zu optimieren (dichte Supervision) | end-to-end, sub-pixel, kein Quantisierungsfehler, schnell/klein |
| Schwächen | Quantisierung durch Output-Stride, teurer (hochauflösende Feature-Maps), argmax nicht differenzierbar | unimodal, verliert räumliche Struktur, historisch schwerer zu trainieren |
| Vertreter | Hourglass, SimpleBaseline, HRNet, MoveNet (Refinement) | DeepPose, BlazePose (Inferenz), YOLOv8-Pose |

**Einordnung der drei Thesis-Modelle:** MediaPipe = Regression zur Inferenz (Heatmaps
nur im Training); MoveNet = hybrid (Regression zur Zuordnung + Heatmap zur Präzision);
YOLOv8 = pure Regression im Detection-Head. → Erklärt qualitativ unterschiedliche
Fehlerbilder (z. B. Keypoint-Displacement vs. Confusion) — als Hypothese nutzbar.

## 5. Klassiker-Timeline (je 1 Satz, chronologisch erzählbar)

1. **DeepPose (2014):** erste DNN-Regression, kaskadierte Verfeinerung — Start der Ära.
2. **Stacked Hourglass (2016):** wiederholte Encoder-Decoder-Module mit intermediate
   supervision, Heatmaps — dominantes Single-Person-Design.
3. **OpenPose / PAF (2017/2019):** bottom-up; Part Affinity Fields = 2D-Vektorfelder
   entlang der Limbs, bipartites Matching fürs Grouping — Echtzeit-Multi-Person.
4. **SimpleBaseline (2018):** ResNet + 3 Deconv-Schichten schlägt komplexe Designs —
   "simplicity works".
5. **HRNet (2019):** parallele Multi-Resolution-Zweige, Hochauflösung durchgehend —
   langjähriger top-down-SOTA.
6. **Mobile-Ära (2020+):** BlazePose, MoveNet, YOLO-Pose — Fokus auf
   Latenz/Edge-Deployment statt Benchmark-SOTA. **Die Thesis evaluiert genau diese
   Klasse unter Deployment-Bedingungen.**

## 6. Metrik-Formeln (Drill)

- **OKS = exp(−dᵢ²/(2·s²·kᵢ²))**, s = √(Segmentfläche), kᵢ per-keypoint-Konstante;
  COCO-AP = Mittel über OKS-Schwellen 0.50:0.05:0.95.
- **PCKh@0.5**: korrekt, wenn Distanz < 0.5 · Kopfsegmentlänge (MPII).
- **MPJPE**: (1/N)·Σ‖p̂ᵢ − pᵢ‖₂.
- **NMPJPE (Thesis)**: MPJPE / Referenz-Körpersegmentlänge (exakte Definition Section
  3.5 — bei Kapitel 3 verifizieren und hier eintragen). ⚠️ Nicht mit dem
  3D-NMPJPE (scale-aligned) aus der Human3.6M-Literatur verwechseln — bei Nachfrage
  aktiv abgrenzen.

## 7. Erwartbare Gall-Niveau-Fragen (Kurzantworten drillen)

1. *"Sketch the MoveNet decoding step — how does it avoid picking up the coach's
   keypoints?"* → Keypoint-Heatmap-Maxima werden inverse-distance-gewichtet zur
   regressierten Initialposition → Maxima nahe der Zielperson gewinnen. (Und trotzdem
   zeigt die Thesis empirisch, wo das versagt → Stärke der Arbeit!)
2. *"Why do all three models struggle with lateral views?"* → Selbst-Okklusion:
   kontralaterale Joints verdeckt; Trainingsdaten-Bias Richtung frontal; bei 2D-GT
   zusätzlich Projektions-Ambiguität. (Thesis: Rotation-Sensitivity-Analyse.)
3. *"What's the output stride problem?"* → Heatmap-Auflösung = Input/Stride;
   Quantisierungsfehler ∝ Stride → deshalb Offset-Heads (MoveNet) bzw. Sub-Pixel-
   Regression (BlazePose).
4. *"How would you extend this to 3D joint angles for actual physiotherapy metrics?"*
   → 2D-Lifting (z. B. VideoPose3D-artig) oder direkte 3D-Modelle (MediaPipe liefert
   world landmarks); braucht Kalibrierung/Skalenreferenz; Future Work — bewusst
   außerhalb des Scopes, weil 2D der gemeinsame Nenner der Modellklasse ist.
5. *"Your GT comes from projected MoCap — what error sources does that introduce?"* →
   Marker-zu-Joint-Offset (Skelett-Modell), Pinhole-Projektion mit geschätzten
   Extrinsics (Maßband + Optimierung laut Dataset-Paper), Synchronisation — deshalb
   NMPJPE-Level nicht als absolute Wahrheit, sondern als faire Vergleichsbasis über
   Modelle framen (alle Modelle sehen dieselbe GT).
6. *"Why not fine-tune the models on REHAB24-6?"* → Ziel ist Off-the-shelf-Deployment-
   Evaluation (was ein App-Entwickler heute nutzen kann); Fine-Tuning wäre eine andere
   Fragestellung + 10 Subjects zu klein für sauberes Split-Design.
