# DEFENSE CHEAT SHEET — ausdrucken & mitbringen

## 1. Kernzahlen (die Pflicht-20)

| Dimension | MediaPipe | MoveNet | YOLOv8 | Gewinner |
|---|---|---|---|---|
| NMPJPE (valid, mean) | 12.5% | **10.5%** | 12.8% | MoveNet |
| Displacement (% Torso) | 6.0% | **3.8%** | 5.2% | MoveNet |
| Rotation frontal→lateral | +28% | **+18%** | +32% | MoveNet |
| Volle 12/12-Skelette | 55% | 38% | **79%** | YOLOv8 |
| Coach-Failure-Rate | **9.1%** | 13.8% | 13.9% | MediaPipe |
| Coach: 2. Person gemeldet | 23% | 16% | 62% | — |
| FPS (CPU, Haupt-Konfig) | 14.7 | **27.7** | 18.9 | MoveNet |

- Statistik: MoveNet gewinnt **alle 10 Cluster** gegen beide (p = 2/1024 = Minimum);
  alle 3 Paare Holm-signifikant; MP−MoveNet +1.01pp [0.62, 1.42].
- Failure-Signaturen: MP **Keypoint-Displacement** 69.6% (von 1,583) · MoveNet
  **Confidence-Collapse** 51.4% (von 1,875) · YOLO **Missing-Detection** 63.8%
  (von 5,610; ≈20× MPs No-Detection-Rate).
- Varianten-Extreme: SP_Lightning 100.6 FPS / 15.0% NMPJPE · YOLO_m 10.9% / 4.6 FPS ·
  MP_Heavy 11.5% / 6.0 FPS. **MultiPose dominiert alle MP/YOLO-Varianten in
  Accuracy UND Speed** (Pareto-Front = MultiPose + SP_Lightning).
- COCO: REHAB-#1 → COCO-#4; COCO-#1 = YOLO_Medium; MP stürzt ab (5→8);
  Hauptmodelle: MoveNet überall #1, MP/YOLO tauschen.
- Hips: MP 16.7% vs. MoveNet 11.3%; ohne Hips schrumpft Gap 2.01→1.21pp (~40%).
- Dataset: 10 Subjects (6m/4w, 25–50) · 6 Übungen · 63/65 Recordings (die 2 übrigen =
  **PM_117a/b**, eine in zwei Teile gesplittete Ex5-Session außerhalb des Standard-
  Namensschemas) → 126 Sequenzen · 1,072 Reps · GT: 41 Marker → 26 Joints → 2D ·
  12 evaluierte Joints · jeder 3. Frame = 10 Hz · 367,200 → 363,074 → 354,879 Frames ·
  COCO: 1,519 Bilder · Coach: 5 Videos (PM_010/011/108/119/121), 1,178 Frames ·
  PM_010 bricht ALLE drei (>65%).

## 2. Jede Entscheidung + 1-Zeilen-Verteidigung

| Entscheidung | Verteidigung |
|---|---|
| Diese 3 Modelle | CPU-Echtzeit + reife Ökosysteme + **1 pro Paradigma** (Verhalten→Architektur attribuierbar); Černek testete MP+YOLO, nie MoveNet |
| Nicht RTMPose/ViTPose/OpenPose | RTMPose: top-down-Slot besetzt, weniger Mobile-Reife; ViTPose: server-grade; OpenPose: unmaintained, CPU-langsam |
| Kein Fine-Tuning | MP+MoveNet = frozen TFLite ohne Trainings-Pipeline (nur YOLO tunebar = unfair); N=10 zu klein für Split; off-the-shelf IST die Deployment-Frage |
| 2D statt 3D | Nur MP hat natives 3D (hüft-zentriert) → 3D-Vergleich unmöglich ohne Lifting-Confound; 2D = gemeinsamer Nenner |
| Proxy-Benchmark (healthy) | Ziel = Joint-Lokalisierung, nicht Übungsbewertung; falsche Ausführungen im Datensatz → Bewegungsvielfalt abgedeckt; Lücke = Patienten-Appearance (Alter, Kleidung, Hilfsmittel) → Future Work #1 |
| Frame-independent | Modelle, nicht Pipelines; YOLO hat kein natives Tracking → alles andere unfair; Werte = upper bound, Deployment nur besser |
| 12 Joints | kleinster anatomisch korrespondierender Nenner (33/17/26); GT hat keine Gesichts-Marker |
| NMPJPE statt PCK/OKS | PCK verschluckt Fehlergröße; OKS braucht COCO-Sigmas + Segmentfläche (GT hat beides nicht); NMPJPE = volle Verteilung, skalenfrei, interpretierbar |
| Joint-Filter 0.5 | Modelle liefern Konfidenz genau dafür; joint-level = datenerhaltend; identisch für alle; MP-Detection-0.1 = nur Eintrittshürde (Default 0.5 erzeugte vermeidbare Detection-Failures) |
| Outlier >100% Torso raus | anderer Fehlermechanismus (falsche Person) — nicht gelöscht, sondern UMGEBUCHT in Failure-Analyse; zwei Regimes, beide berichtet |
| Person-Selection largest | Deployment-Realismus (App hat keine GT); native Outputs; Pseudo-Box würde mit Armspreizung pulsieren; nicht perfekt — Arbeit zeigt selbst den Fehlfall (Panel c) |
| Cluster + Permutation | Frames nicht unabhängig → Person = Einheit; n=10 → keine Verteilungsannahmen haltbar → exakter Test (1024); Holm = wie Bonferroni, mehr Power; Mixed Model bei N=10 fragil |
| CPU-only Desktop | reproduzierbar, ein Gerät, identisch für alle 9; Claim = relative Ordnung; Handy-Benchmark mit Delegates = Future Work |
| Primary-Wahl (Full/MultiPose/Nano) | nicht nach Namensstufe, sondern Latenz-Klasse (14.7/27.7/18.9 FPS) + Vendor-Defaults; MultiPose = einzige protokollfähige MoveNet-Variante; Varianten-Tabelle macht alles transparent |

## 3. Related Work — 1-Zeiler (bei Nachfrage: Toolkit-Satz 2)

- **Černek 2025** (REHAB24-6-Paper, WICHTIGSTES): **Dataset-Paper** (Beitrag =
  Ressource) vs. meine **Evaluationsstudie** (Beitrag = Deployment-Antwort). 5 Deltas:
  (1) + MoveNet — **das ändert den Gewinner und damit die Empfehlung**; (2) Paradigmen
  systematisch (1 pro Klasse) → erst das ermöglicht Drei-Profile + Failure-Signaturen;
  (3) 6 Dimensionen unter einem Protokoll statt Accuracy-Fokus; (4) Multi-Person-
  Dekomposition + Coach = Kernbefund, dort ohne Gegenstück; (5) Inferenzstatistik
  (Cluster/Permutation/Holm/Bootstrap). Beleg: Gap-Tabelle 2.1 (Černek 2/7 Spalten).
- **Rode 2025:** mehr Modelle für klinische Analyse, Fokus per-joint — ich schmaler
  aber tiefer (Deployment-Dimensionen) → komplementär.
- **Aguilar-Ortega 2023 (UCO):** Multi-View-Reha-Datensatz, frontal→lateral schlechter
  — aber server-grade Modelle.
- **Baldinger 2025:** 4 Winkel, OpenPose, Lunges — lateral schlechter; nur 1 Modell.
- **Jo & Kim 2022 / Chung 2022:** Mobile-Vergleiche auf Standard-Daten, Effizienzfokus.
- **Roggio 2024:** Review — hält MP/MoveNet für Reha geeignet, quantifiziert aber
  keine Okklusions-/Crowd-Degradation.
- **Hii 2023:** MediaPipe für Ganganalyse validiert — kein Modellvergleich.
- **Debnath 2022:** Survey — Engpass: fehlende öffentliche Reha-Datensätze.
- **Ullah 2025:** MediaPipe-Physio-App — Anwendung, kein Vergleich.
- **Zheng 2023:** HPE-Survey (Quelle der Paradigmen-Taxonomie). **Zeng 2022
  (SmoothNet):** temporale Glättung als Forschungsfeld → motiviert Jitter-Metrik.
- Klassiker: **DeepPose 14** erste DL-Regression · **Hourglass 16** Heatmaps ·
  **OpenPose 17** PAFs/bottom-up · **SimpleBaseline 18** Einfachheit reicht ·
  **HRNet 19** High-Resolution top-down SOTA.

## 4. Backup-Folien (aktiv anbieten: "I have a slide on that")

**B1** Statistik-Details · **B2** NMPJPE-Formel + Alternativen · **B3** Rotation/65°-
Kalibrierung · **B4** Keypoint-Mapping · **B5** alle 9 Varianten (NMPJPE+FPS) ·
**B6** COCO-Protokoll · **B7** Coach pro Video (PM_010!) · **B8** Per-Joint/Hips ·
**B9** Architektur-Interna (BlazePose/MoveNet/YOLO)

## 5. RQ-Antworten in je 1 Zeile

1. **Accuracy/Completeness:** MoveNet genauester (alle Ebenen, Holm) — aber
   Completeness gehört YOLO → getrennte Dimensionen.
2. **Stability:** MoveNet, ganze Familie Top 3.
3. **Robustness:** bedingungsabhängig — Viewpoint: MoveNet · breites Subset: keiner
   degradiert · Coach: MediaPipe. Kein Einheits-Skalar!
4. **Transfer:** limited, not absent — Leader wechselt, MoveNet bleibt bei den
   Hauptmodellen vorn; protokollgebunden.
5. **Deployment:** Default MoveNet MultiPose; YOLO bei Completeness, MP bei
   Coach/Landmarks, Lightning bei Speed; Deployment-Guidance ≈ so wichtig wie Modellwahl.

## 6. Die 3 Hypothesen — IMMER auf diesem Level halten

- **Failure ↔ Paradigma:** "observational, one family per paradigm — I can't separate
  architecture from training data. Consistent, but a hypothesis."
- **Hip = Landmark-Definitions-Mismatch:** "error concentrated in one region, removing
  hips closes 40% of the gap — consistent with a definition mismatch; evidence is
  indirect. The index mapping itself is validated."
- **Coach-Mechanismus:** "sits at the selection stage — candidate retention,
  thresholding — internals not observable, deliberately left open."

## 7. Notfall-Sätze (Nicht-Wissens-Toolkit)

- Zahl weg: *"I don't have the exact number in my head — it's in Table X. The order
  of magnitude was around …"*
- Paper-Detail weg: *"I'd have to check the paper for the specifics — what I took
  from it for my thesis was …"*
- Design-Detail unsicher: *"Let me reconstruct that: the constraint was …, so the
  choice was …"*
- Keine Ahnung: *"Fair question — I don't know with confidence. My expectation would
  be …, because …, and here is how I would test it."*
- Frage unklar: *"Just to make sure I answer the right thing — do you mean … or …?"*
- Denkpause: *"That's a good question — let me take that apart."*

## 8b. Universal-Schema für JEDE unvorbereitete „Warum nicht X?"-Frage

**Die 5 Anker der Arbeit:** (1) Fairness über die 3 Modelle · (2) Deployment-Realismus
· (3) Modell- statt Pipeline-Isolation · (4) Reproduzierbarkeit/Transparenz ·
(5) ehrliche Statistik bei kleinem N.

**4 Schritte, laut gedacht:**
1. Anker nennen: "The goal at that step was …"
2. X live bewerten: "X would give me …, but it would cost/require …"
3. Ehrlich verorten (NIE bluffen): considered+rejected / not on my radar — evaluating
   it now / genuine alternative, good extension.
4. Konsequenz: "Would it change the main conclusion? My expectation is …, because …"

**Standard-Satz:** "That's a fair alternative — I didn't evaluate it at the time.
Thinking it through against my criteria: it would …, but it would … . My expectation
is that it wouldn't change the main conclusion, because it affects all three models
equally. It would be a natural extension."

**Scope erst NACH Schritt 2 nennen** (allein = Ausrede; nach Live-Bewertung = Entscheidung).

**Zweitprüfer Krüger (MoCap-Experte) — wahrscheinliche GT-Fragen:** GT-Fehlerquellen
(Marker-zu-Gelenk-Offset, Projektion mit geschätzten Extrinsics, Sync) → "that's why I
treat NMPJPE levels as a fair comparison basis, not absolute truth — all models see
the same ground truth" · Skelett-Konventionen → B4 · warum nicht 3D-GT → A3.

## 8. Gefährlichste Fragen → wo die Antwort steht

Modellwahl/RTMPose → §2 Z.1–2 · Fine-Tuning → §2 Z.3 · Healthy Volunteers → §2 Z.5 ·
Frame-independent → §2 Z.6 · Statistik-Tiefe → §2 Z.12 + B1 · Hip → §6 + B8 (+
Trainingsdaten: MoveNet/YOLO auf COCO-Hüften trainiert, BlazePose eigenes Schema, GT =
MoCap-Gelenkzentrum → 3 Konventionen) · Handy-30-FPS → GPU-Delegate + Tracking-Modus
vs. mein CPU-only + IMAGE-Modus; monotone Latenz-Treppen = Beleg sauberer Messung ·
Paradigma-Confound → §6 Z.1 · „5 Coach-Videos genug?" → benannte Limitation,
deskriptiv, PM_010-Gegenprobe · „Warum nicht einfach MoveNet krönen?" → Deployment
ist mehrkriteriell, drei Profile SIND die Antwort.
