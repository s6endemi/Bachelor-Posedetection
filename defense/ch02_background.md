# Chapter 2 — Background and Related Work (Verteidigungsanalyse)

> Quelle: `thesis/chapters/02_background.tex` (final, vollständig gelesen)
> **Gall-Territorium: höchste Priorität.** Ergänzend: `fundamentals_drill.md`.

## 1. Kernaussagen

1. **Zwei Grundunterscheidungen:** keypoint-based vs. mesh-based; 2D image-coordinate
   vs. 3D world-coordinate. Diese Arbeit: strikt 2D, keypoint-based, gegen projizierte
   MoCap-GT.
2. **Drei Architektur-Paradigmen** organisieren den Vergleich:
   - **Top-down:** erst Person detektieren (BBox/ROI), dann Pose pro Region. Kosten
     skalieren mit Personenzahl; Genauigkeit hängt am Upstream-Detektor.
     (Zitiert: DeepPose, Stacked Hourglass, SimpleBaseline, HRNet.)
   - **Bottom-up:** alle Keypoints im ganzen Bild in einem Pass, dann Gruppierung zu
     Personen. Kanonisch: OpenPose (Part Affinity Fields). Fixe Kosten bzgl.
     Personenzahl; Schwachstelle Grouping bei räumlich nahen Personen.
   - **One-stage:** Person-Detection + Keypoints in einem Forward-Pass (Detektor mit
     Pose-Head). Guter Speed-Accuracy-Trade-off. Beispiel: YOLOv8-Pose.
   - Zuordnung der Thesis: **MediaPipe = top-down, MoveNet = bottom-up, YOLOv8 = one-stage.**
3. **Modelle:** MediaPipe Pose/BlazePose (33 Landmarks, Lite/Full/Heavy, TFLite);
   MoveNet (17 COCO-Keypoints, MultiPose Lightning als Haupt-Konfiguration — bis zu 6
   Personen + BBoxen; SinglePose Lightning/Thunder in der Familienanalyse); YOLOv8-Pose
   (17 COCO-Keypoints, Nano als Haupt-Konfiguration, Small/Medium in Familienanalyse,
   Ultralytics-PyTorch-Runtime statt TFLite).
4. **Metriken:** PCK, OKS, MPJPE, NMPJPE definiert; Thesis nutzt **NMPJPE** (skalen-
   normalisiert auf 12-Joint-Skelett) + normalized frame-to-frame prediction
   displacement als Stabilitäts-Proxy (Motivation über SmoothNet, Zeng 2022).
5. **Research Gap — 6 Lücken:** (1) MoveNet fehlt auf Reha-Benchmarks, (2) temporale
   Stabilität selten als Video-Metrik, (3) Multi-Person als ein Regime behandelt statt
   dekomponiert, (4) Intra-Familien-Trade-offs unvollständig, (5) Cross-Dataset-Transfer
   angenommen statt getestet, (6) Abhängigkeitsstruktur wiederholter Messungen selten
   inferenziell modelliert. + Research-Gap-Tabelle 2.1 (7 Dimensionen).

## 2. ⚠️ Die zwei Fallen dieses Kapitels

### Falle 1: "Is MoveNet really bottom-up?" (Taxonomie-Angriff)

MoveNet macht **kein OpenPose-artiges Keypoint-Grouping**. Es ist CenterNet-artig:
Person-Center-Heatmap + Keypoint-Regression + Keypoint-Heatmap-Refinement + Offsets,
alles in einem Pass. Gall kann die bottom-up-Einordnung challengen.

**Verteidigung (EN):**
> "MoveNet follows the CenterNet family: it predicts a person-center heatmap and
> regresses keypoints from the center, refined by full-image keypoint heatmaps — all in
> a single pass over the full image, without a person detector or per-person crops. Its
> cost is independent of person count, which is the defining operational property of
> bottom-up methods, and Google itself classifies it as bottom-up. I acknowledge the
> taxonomy is fluid — center-based methods are sometimes labeled 'single-stage' — but
> the operational distinction I rely on is: per-person crop (top-down) vs. full-image
> single pass (MoveNet) vs. detector-with-pose-head (YOLO)."

Kernpunkt: **Nicht auf Label-Ebene streiten, sondern die tatsächliche Architektur
beschreiben können.** Wer die Mechanik erklärt, dem verzeiht man jedes Label.

### Falle 2: "Your gap table is constructed so only you tick all boxes."

Klassischer Einwand gegen jede Research-Gap-Tabelle.

**Verteidigung (EN):**
> "The seven dimensions aren't chosen post hoc — they follow from the four deployment
> constraints in the introduction: home use implies viewpoint variation, other people in
> the room, video feedback, and CPU-only hardware. Each constraint maps to a dimension;
> variants and cross-dataset transfer address deployment choice and generalization. The
> table documents coverage, not quality — the cited studies are strong within their own
> scope."

## 3. Related Work — 1-Satz-Steckbriefe (müssen sitzen!)

| Paper | 1 Satz |
|---|---|
| **Černek et al. 2025** (REHAB24-6) | Führen REHAB24-6 ein und evaluieren u. a. MediaPipe + YOLOv8-Varianten darauf — **aber kein MoveNet, kein Jitter, keine Coach-Analyse** → direktester Anker + Abgrenzung. |
| **Rode et al. 2025** | Breiter Modellvergleich für klinische Bewegungsanalyse, Fokus per-joint accuracy — keine Multi-Person-/Stabilitäts-/Transfer-Analyse. |
| **Debnath et al. 2022** | Survey CV-basierte Reha-Assessment; zentraler Engpass: Mangel öffentlicher Reha-Datensätze. |
| **Ullah et al. 2025** | MediaPipe-basiertes Physio-Scoring-System — Anwendungsfall, kein Modellvergleich. |
| **Aguilar-Ortega et al. 2023** (UCO) | Multi-Viewpoint-Reha-Datensatz; deutlicher Fehleranstieg frontal→lateral — aber server-grade Modelle. |
| **Baldinger et al. 2025** | 4 Kamerawinkel, OpenPose, Lunges; höhere Fehler lateral — Einzelmodell, nicht mobile. |
| **Jo & Kim 2022** | OpenPose/PoseNet/MoveNet auf Mobilgeräten — Effizienzfokus, Standard-Daten. |
| **Chung et al. 2022** | Breiterer Mobile-Vergleich, ebenfalls Effizienzfokus. |
| **Roggio et al. 2024** | Review; hält MediaPipe/MoveNet für remote fitness/rehab geeignet — quantifiziert aber keine Degradation bei Okklusion/Crowds. |
| **Hii et al. 2023** | MediaPipe für Ganganalyse validiert — ohne Modellvergleich. |
| **Zheng et al. 2023** | Standard-Survey Deep-Learning-HPE (Taxonomie-Quelle). |
| **Zeng et al. 2022** (SmoothNet) | Temporale Glättung als eigenes Forschungsthema → motiviert Jitter als Metrik. |

Architektur-Klassiker (zitiert in 2.2 — je 1 Satz können):
- **DeepPose (Toshev & Szegedy 2014):** erste Deep-Learning-Pose-Regression (direkte Koordinaten-Regression, kaskadiert).
- **Stacked Hourglass (Newell et al. 2016):** wiederholtes Down-/Upsampling mit intermediate supervision, Heatmap-basiert.
- **SimpleBaseline (Xiao et al. 2018):** ResNet + Deconv-Layer — zeigt, dass einfache Architektur reicht.
- **HRNet (Sun et al. 2019):** hält Hochauflösungs-Repräsentation parallel durch — lange SOTA top-down.
- **OpenPose (Cao et al. 2019):** bottom-up mit Part Affinity Fields (2D-Vektorfelder für Limb-Zugehörigkeit) für das Grouping.

## 4. Metriken — exakte Definitionen (Drill)

- **PCK:** Anteil Keypoints innerhalb Schwellwert-Distanz zur GT; normalisiert z. B. auf
  Kopfgröße (**PCKh@0.5**, MPII-Standard) oder Torso.
- **OKS** (COCO): per-Keypoint-Ähnlichkeit, OKS = exp(−d²/(2·s²·k²)), s = Objekt-Scale
  (√Fläche), k = keypoint-spezifische Konstante (Augen klein, Hüften groß); AP über
  OKS-Schwellen 0.5–0.95 gemittelt.
- **MPJPE** (Human3.6M): mittlere euklidische Distanz Prediction↔GT pro Joint (mm/px).
- **NMPJPE (Thesis-Definition!):** MPJPE normalisiert durch eine Referenz-
  Körpersegmentlänge → skalenfreier Prozentwert. ⚠️ In der 3D-Literatur (Human3.6M)
  bezeichnet NMPJPE oft scale-aligned MPJPE (Prediction in Scale zur GT ausgerichtet) —
  falls Gall das anmerkt: Thesis-Definition wird in Section 3.5 explizit definiert;
  Normalisierung durch Körpersegment ist die 2D-übliche Variante (verwandt mit
  PCK-Normalisierung). Genaue Definition in Kap.-3-Analyse prüfen.
- **Benchmarks:** COCO (~200k Bilder, 17 Keypoints, OKS/AP; val2017 = 5k Bilder), MPII
  (~25k Bilder, 16 Joints, PCKh), Human3.6M (3.6M Frames, MoCap, MPJPE, 3D-kanonisch).

## 5. Design-Entscheidungen + Verteidigung

| Entscheidung | Ehrlich | Strategisch |
|---|---|---|
| Paradigmen als Organisationsprinzip | natürliche Struktur | macht Auswahl systematisch: 1 Modell pro Paradigma → Unterschiede architektonisch interpretierbar (Kap. 6 Drei-Profile) |
| Modell-Beschreibungen in 2.3 knapp | Platz | Details stehen dort, wo sie gebraucht werden (Kap. 3 Konfiguration); Tiefe gehört in die Verteidigung, nicht in den BA-Text |
| NMPJPE statt OKS/PCK als Primärmetrik | MoCap-GT gibt kontinuierliche Distanzen her | OKS braucht per-keypoint sigmas + Objekt-Scale (COCO-spezifisch); PCK verschluckt Fehlergrößen oberhalb/unterhalb des Schwellwerts; NMPJPE ist kontinuierlich, skalenfrei, klinisch interpretierbar |
| MultiPose Lightning als Haupt-MoveNet | Multi-Person-Setting erfordert Multi-Person-Fähigkeit | einzige MoveNet-Variante mit nativen BBoxen + bis zu 6 Personen → faire Vergleichbarkeit der Person-Selection |
| YOLOv8n als Haupt-YOLO | mobile-oriented entry point | konsistent mit Mobile-Deployment-Klasse der anderen Haupt-Modelle |

## 6. Gefährliche Fragen + Musterantworten (EN)

**Q1: "Explain how BlazePose actually works."** → siehe fundamentals_drill.md (Detector-
Tracker, Face-anchored Detector, Heatmap+Regression-Training, 33 Landmarks).

**Q2: "Why NMPJPE and not OKS, given COCO uses OKS?"**
> "OKS encodes COCO-specific per-keypoint tolerances and needs an object scale from
> segmentation area — neither is defined for my MoCap ground truth. PCK collapses errors
> into hit-or-miss around a threshold. NMPJPE keeps the full error distribution,
> normalizes by a body-segment length so subjects of different size are comparable, and
> is directly interpretable as a fraction of body size — which is what matters for
> judging clinical usability."

**Q3: "MoveNet is bottom-up — where is the grouping step?"** → Falle 1 oben.

**Q4: "Why is OpenPose not in your evaluation if it's the canonical bottom-up model?"**
> "OpenPose is the canonical reference architecturally, but it's no longer maintained
> and its CPU throughput is far below real-time — it fails the deployment constraint
> that defines my model class. MoveNet covers the bottom-up slot with a
> mobile-optimized design."

**Q5: "Which prior work is closest to yours, and what exactly do you add?"**
> "Černek et al. 2025 — the REHAB24-6 paper itself. They evaluate MediaPipe and
> YOLOv8-Pose variants on it. I add MoveNet — absent from all rehabilitation benchmarks
> I reviewed — plus temporal displacement, the two-subset multi-person decomposition
> including the coach extreme case, family variants under one protocol, cross-dataset
> ranking transfer to COCO, and subject-cluster inference."

**Q6: "Human3.6M is MoCap-based video — why not use it instead of REHAB24-6?"**
> "Human3.6M is 3D-canonical and covers everyday actions like walking, sitting,
> discussing — not guided rehabilitation exercises, no therapist in frame, and no
> exercise-correctness structure. REHAB24-6 matches the deployment setting:
> physiotherapist-guided exercises, fixed cameras, repeated recordings per subject."

## 6b. Verständnis-Klärungen (Session 7.7.2026)

- **Keypoint- vs. mesh-based:** Keypoint = Skelett aus diskreten Punkten (Koordinaten-
  Liste; 17 COCO / 33 MP) — leicht, liefert Gelenke für Winkel/Bewegung. Mesh = ganze
  Körperoberfläche als 3D-Dreiecksnetz (SMPL, ~6.890 Vertices; Form + Pose) — für
  Avatare/Körperform, zu schwer für Mobile. Thesis: keypoint-based, weil Reha Gelenke
  braucht und die deploybare Mobile-Klasse durchweg keypoint-based ist.
- **Bounding Box:** kleinstes umschließendes Rechteck (x, y, w, h + Confidence) =
  Standard-Output der Objektdetektion. Drei Rollen in der Thesis: (1) top-down-
  Baustein (Box → Crop → Pose), (2) Person-Selection via größter Box-Fläche
  (≈ prominenteste Person = Patient), (3) MediaPipe hat KEINE nativen Boxen →
  Torso-Extent-Heuristik statt Pseudo-Box aus Keypoint-Extremen (schwankt mit
  Armspreizung).
- **Drei Architekturen als Merkbild:** Top-down „erst ausschneiden, dann messen"
  (Kosten ∝ Personenzahl, hängt am Detektor, pro Person hochauflösend). Bottom-up
  „erst alle Punkte, dann puzzeln" (ein Pass, kostenkonstant; Grouping fragil bei
  nahen Personen; MoveNet = center-basiert: Zentren-Heatmap → Regression →
  Heatmap-Refinement). One-stage „alles auf einmal" (Detektor mit Pose-Head, Box +
  17 Keypoints pro Detektion). *Detect→crop→estimate / all keypoints→group /
  detect+estimate in one shot.*
- **Displacement-Satz zerlegt:** frame-to-frame displacement = euklidische Distanz
  derselben Gelenk-Vorhersage zwischen t und t+1; normalisiert durch GT-Torso-Länge
  (Pixel → % Körpergröße, vergleichbar über Distanz/Auflösung); unsmoothed = Roh-Spur
  ohne Filter. Praxis: hohes Displacement = zappelndes Live-Skelett trotz ruhigem
  Patienten.
- **Top-down vs. bottom-up — das Crop-Kriterium:** Entscheidend ist nicht, OB etwas
  detektiert wird, sondern ob ein **Per-Person-Crop mit zweitem Netz** existiert.
  MediaPipe: Netz 1 (Detektor) → Ausschnitt → Netz 2 (Pose) sieht nur die Person =
  top-down. MoveNet: EIN Netz, ein Full-Image-Pass; Center-Heatmap, Regression und
  Keypoint-Heatmaps entstehen aus denselben Features, kein Crop = bottom-up.
  Merksatz: *Top-down = das Pose-Netz sieht eine Person; bottom-up = das Netz sieht
  immer das ganze Bild.*
- **⚠️ Face-Filter-Korrektur (gegen finale Arbeit geprüft, 7.7.2026):** Die finale
  Thesis claimt KEINEN face-basierten Filter-Mechanismus. Kap. 6.3 lässt den
  Coach-Mechanismus explizit offen (Kandidaten: candidate retention, thresholding,
  post-detection selection; "cannot be localized further"). BlazePose-face-Detector =
  Q&A-Hintergrundwissen (Paper-Fakt), als ungetestete konsistente Hypothese
  kennzeichnen → qa_catalog.md B7.
- **Displacement ≠ GT-Vergleich:** Displacement vergleicht Prediction(t) mit
  Prediction(t+1) — Modell mit sich selbst. GT liefert nur die Torso-Länge als Lineal
  (Normalisierung). Wackelkamera-Analogie: drei Kameras, dasselbe Auto — Unterschiede
  im Gesamtwackeln = Stativqualität, echte Bewegung muss man nie kennen.
- **„Wie kann man temporal stability messen, wenn frame-independent?"** (Eigene
  Entdeckung — prüfungsrelevant!) Auflösung: (1) Stabilität ist Eigenschaft der
  OUTPUT-FOLGE, kein Modell-Mechanismus: fast identische Inputs (konsekutive Frames)
  sollten fast identische Outputs geben — gemessen wird Modell-Empfindlichkeit.
  (2) „Proxy", weil Messwert = echte Bewegung + Modellzittern, nicht trennbar; aber
  echte Bewegung ist über Modelle IDENTISCH (gleiche Frame-Paare) → kürzt sich im
  Modellvergleich raus. Frame-independent macht die Aussage STÄRKER: inhärente
  Stabilität, nicht Filter-Verdienst. EN-Antwort → qa_catalog.md B6.

## 7. Folien-Kandidaten aus Kapitel 2

- **Folie "Three architectural paradigms":** Diagramm top-down vs. bottom-up vs.
  one-stage (je 1 Mini-Skizze) + Zuordnung der 3 Modelle. **Pflichtfolie** — sie trägt
  die spätere Drei-Profile-Synthese.
- **Folie "Research gap":** Tabelle 2.1 vereinfacht (max. 4–5 Zeilen) ODER als
  1-Satz-Statement + "no prior work combines these dimensions".
- Metriken NICHT als eigene Folie — NMPJPE-Definition kommt kompakt bei Methodology.
