# Chapter 3 — Methodology (Verteidigungsanalyse)

> Quelle: `thesis/chapters/03_methodology.tex` (final, vollständig gelesen)
> **Höchste Angriffsfläche der Arbeit. Jede Zahl und jede Entscheidung hier muss sitzen.**

## 1. Kernaussagen

1. **Framing-Satz (Zeile 1 des Kapitels, wörtlich verinnerlichen):** *"The study is
   motivated by home-based rehabilitation, but it is designed as a controlled model
   comparison on a rehabilitation-oriented proxy benchmark, not as direct clinical
   validation."* → Das ist der Schutzschild gegen jede "aber das sind keine Patienten"-Frage.
2. **REHAB24-6:** 10 gesunde Erwachsene (6m/4w, 25–50), 6 Übungen, 65 öffentliche
   Recordings → **63 verwendete Recording-IDs = 126 Camera-View-Sequenzen**, 1.072
   Repetitionen, 2 synchrone RGB-Kameras à 30 fps, GT = 41-Marker-MoCap → 26-Joint-
   Skelett, 2D-projiziert. **367.200 model-frame observations vor Filterung.**
   c17 ≈ frontal, c18 ≈ lateral.
3. **COCO val2017 auxiliary:** 1.519 Bilder (≥1 Person mit ≥6 gelabelten Keypoints),
   Matching = nächstes Torso-Center zur GT, nur sichtbare GT-Joints, gleiche 12 Joints —
   **explizit anderes Protokoll** → nur Accuracy/Ranking, deskriptiv.
4. **Konfigurationen:** Primär MediaPipe **Full** (complexity=1,
   `min_pose_detection_confidence=0.1`!), MoveNet **MultiPose**, YOLOv8 **Nano**.
   Varianten: MP Lite/Heavy, MoveNet SP Lightning/Thunder, YOLOv8 Small/Medium.
5. **12-Joint-Mapping:** Schultern, Ellbogen, Handgelenke, Hüften, Knie, Knöchel
   (33 MP / 17 COCO / 26 GT → 12 gemeinsame). Gesicht raus (keine GT-Marker).
6. **NMPJPE** = mittlerer Joint-Fehler / **Torso-Länge** (GT-Schulter-Mittelpunkt ↔
   GT-Hüft-Mittelpunkt) × 100%.
7. **Rotation:** aus 3D-GT-Schultern (atan2), **globaler c17-Offset 65° ± 5°** aus 3
   frontalen Referenz-Recordings (PM_114, PM_122, PM_109); θ_c18 = 90° − θ_c17;
   10°-Bins > Kalibrierunsicherheit; Verteilung **bimodal** → "viewpoint sensitivity
   under fixed camera geometry", keine dichte Rotationsstudie.
8. **Multi-Person:** (a) Metadaten-Subset `extra_person` severity ≥ 2; (b) **Coach-Subset:
   5 c17-Sequenzen (PM_010, PM_011, PM_108, PM_119, PM_121), manuell identifiziert.**
9. **Person-Selection:** MoveNet/YOLO = größte BBox-Fläche; MediaPipe = größter
   Torso-Extent (keine nativen Boxen; Pseudo-Boxen aus Keypoint-Extremen bewusst
   vermieden). **Selection = Evaluations-Design, nicht Modell-Eigenschaft.**
10. **Displacement-Metrik:** ‖p(t+1) − p(t)‖ / mittlere GT-Torso-Länge; Joint muss in
    beiden Frames valid sein; **GT-Bewegung wird nicht subtrahiert → Proxy.**
11. **Filterung:** Joint-Level-Confidence < 0.5 raus (nicht Frame-Level!); keine
    Detection = detection failure; NMPJPE > 100% Torso = katastrophaler Outlier.
    **Zwei Regimes: central tendency (gefiltert) vs. failure behavior (Outlier drin).**
12. **Frame-independent:** IMAGE mode / kein Tracking / kein `model.track()` →
    Ergebnisse = **upper-bound failure characteristics** der Roh-Outputs.
13. **Inference-Benchmark:** CPU-only, 3 c17-Videos, 500 Frames + 50 Warm-up,
    sequenziell, native Runtimes (MP Tasks API, MoveNet TFLite, YOLO PyTorch).
14. **Sampling + Statistik:** jeder 3. Frame (10 Hz); Median pro Sequenz → Mittel pro
    Subject-Cluster (`person_id`, **10 Cluster**) → **exakte sign-flip-Permutationstests
    (2¹⁰ = 1024 Vorzeichen)**, **Bootstrap-CIs (20.000 Resamples)**, **Holm-Korrektur**;
    Video-Level nur Sensitivity (Monte Carlo).

## 2. Memorier-Zahlen (Blitz-Abruf)

| Fakt | Wert |
|---|---|
| Subjects / Cluster | 10 (6m, 4w, 25–50 J.) |
| Recordings öffentlich / verwendet | 65 / 63 → 126 Sequenzen |
| Frames vor Filterung | 367.200 model-frame observations |
| Repetitionen | 1.072 |
| Evaluierte Joints | 12 (Arme + Beine, kein Gesicht) |
| Confidence-Filter | 0,5 joint-level; MP Detection-Stage 0,1 |
| Outlier-Grenze | NMPJPE > 100% Torso |
| Sampling | jeder 3. Frame = 10 Hz |
| Rotation-Offset | c17: 65° ± 5° (3 Referenz-Recordings); Bins 10° |
| Coach-Videos | 5 (PM_010, PM_011, PM_108, PM_119, PM_121) |
| Permutationstest | exakt, 2¹⁰ = 1024; Bootstrap 20.000; Holm |
| COCO-Subset | 1.519 Bilder, ≥6 Keypoints, torso-center matching |
| Inference | 3 Videos × 500 Frames + 50 Warm-up, CPU |

## 3. Die sechs heißen Zonen (Fragen + EN-Antworten)

### Zone 1: NMPJPE-Terminologie (Human3.6M-Kollision)
Die Thesis zitiert Ionescu 2014 bei der NMPJPE-Einführung. In der 3D-Literatur ist
"N-MPJPE" aber **scale-aligned MPJPE** (Prediction optimal auf GT-Skala skaliert) —
hier dagegen MPJPE **normalisiert durch Torso-Länge**.
> "I define NMPJPE explicitly in Equation 3.1: mean per-joint error as a percentage of
> ground-truth torso length. I'm aware that in the 3D literature N-MPJPE sometimes
> denotes scale-aligned MPJPE; my usage follows the 2D convention of body-segment
> normalization, analogous to PCK's normalization — it makes errors comparable across
> subjects, camera distances, and resolutions, and directly interpretable as a fraction
> of body size."

### Zone 2: Der 65°-Rotations-Offset (empirische Kalibrierung)
Angriff: "You hand-calibrated camera geometry from three videos?"
> "REHAB24-6 does not publish full camera extrinsics, so I estimated a single global
> camera-relative offset from three recordings where subjects stand approximately
> frontal — with an estimated uncertainty of about ±5 degrees. Two design decisions
> protect the analysis: the orientation comes from 3D ground-truth shoulders, not from
> predictions — avoiding circularity — and the rotation analysis uses 10-degree bins,
> wider than the calibration uncertainty. I also interpret the result as viewpoint
> sensitivity under fixed camera geometry, not as a dense rotation study, because the
> viewpoint distribution is strongly bimodal."
Follow-up *"why atan2 of absolute values?"* → faltet in [0°, 90°], links-/rechts-
symmetrisch; Vorzeichen für Sensitivity irrelevant. *"c18 = 90° − c17?"* → Kameras in
Raumecken, näherungsweise orthogonale Blickachsen; Restfehler steckt in der ±5°/Bin-Toleranz.

### Zone 3: Person-Selection (largest instance) + COCO-Inkonsistenz
Angriff 1: "Why largest bounding box and not highest confidence — or matching to GT?"
> "Deployment realism. A home application has no ground truth to match against — it
> must pick the patient heuristically, and the patient is typically the most prominent
> person in frame. Matching predictions to the ground truth would define away exactly
> the target-retention problem I want to measure. The heuristics follow each model's
> native output — bounding-box area where boxes exist, torso extent for MediaPipe,
> avoiding pseudo-boxes from keypoint extremes that fluctuate with limb spread."
Angriff 2 (die Inkonsistenz-Falle!): "But on COCO you *do* match to ground truth."
> "Deliberately — that's why COCO is labeled an auxiliary benchmark under a different
> protocol. COCO has no designated target person, so target retention is not a
> meaningful concept there; the COCO comparison isolates body-joint accuracy and
> ranking transfer, and I keep it descriptive for exactly this protocol difference."

### Zone 4: Uniformer Confidence-Threshold 0,5
Angriff: "Confidence scores are not calibrated across models — a uniform 0.5 means
different things for each model."
> "Correct, and the thesis handles this in three ways: the threshold is applied at
> joint level, not frame level, so partially reliable predictions stay usable;
> detection completeness is reported as its own dimension, which makes filter effects
> visible per model; and 0.5 is the documented, deployment-typical default. A per-model
> calibration study would be a valuable extension, but it would optimize each model's
> operating point — my goal was a fixed, transparent protocol."
Zusatz: MediaPipe `min_pose_detection_confidence=0.1` ist **Detection-Stage** (stricter
defaults → vermeidbare Detection-Failures), nicht der Joint-Filter — nicht verwechseln!

### Zone 5: Outlier-Entfernung >100%
Angriff: "You delete your worst errors before reporting accuracy."
> "No — I separate two questions. Central-tendency accuracy asks: when the model tracks
> the right person, how precise is it? Catastrophic frames — typically wrong-person
> assignments — would mix a different failure mechanism into that estimate. So they are
> excluded there, but they are *retained and reported* as an outcome in the robustness
> and failure-mode analyses. Nothing is hidden; the two regimes are explicitly defined
> in Section 3.9."

### Zone 6: Statistik — 10 Cluster, Permutation statt Mixed Model
Angriff: "Why not a mixed-effects model for repeated measures?" (Lars' historischer Punkt!)
> "With 10 subjects, a mixed model's variance-component estimates and normality
> assumptions become fragile. I chose the assumption-light route: aggregate to the
> natural independence unit — the subject cluster — and run exact paired sign-flip
> permutation tests over all 1024 sign assignments, with Holm correction and percentile
> bootstrap intervals. The test is exact at this sample size, makes no distributional
> assumptions, and directly respects the dependency structure. Frame- and video-level
> analyses are kept as sensitivity checks; the frame-level ordering is reported in the
> thesis, the video-level result lives in the analysis artifacts — same ranking, same
> significance pattern."
Follow-up *"minimal p-value with 10 clusters?"* → zweiseitig 2/1024 ≈ **0,002** —
zeigen, dass Du das weißt!
Follow-up *"10 Hz sampling — doesn't that alias fast motion in your jitter metric?"*
> "Absolute displacement values depend on the sampling rate, yes — but all models see
> identical frame pairs under the shared protocol, so between-model differences are
> attributable to the models. Rehabilitation movements are slow relative to 10 Hz, and
> the metric is explicitly a stability proxy, not an absolute jitter measurement."
Follow-up *"your displacement metric confounds true motion"* →
> "True motion is identical across models on identical frame pairs — it cancels in the
> comparison. I deliberately do not subtract ground-truth displacement because vector
> decomposition of noise and motion doesn't separate linearly in the norm; the thesis
> names this as a limitation and frames the metric as a proxy."

## 4. Weitere Design-Entscheidungen (Kurzverteidigung)

| Entscheidung | Verteidigung |
|---|---|
| 12 Joints statt 17 COCO | GT hat keine Gesichts-Marker; Reha-relevant sind Extremitäten + Rumpf; kleinster gemeinsamer, anatomisch korrespondierender Nenner |
| Torso-Länge als Normalisierung | GT-basiert (nicht prediction-abhängig), pro Frame, skalenfrei; Foreshortening betrifft alle Modelle identisch → Ranking stabil |
| Frame-independent (IMAGE mode) | Modelle evaluieren, nicht Pipelines; das YOLO-**Netz** hat kein temporales Processing (Ultralytics-Tracking = Pipeline-Schicht, bewusst nicht genutzt) → einheitlich frame-weise = fair; Ergebnisse = upper bound der Roh-Outputs; Kap.-6-Limitation |
| Jeder 3. Frame (10 Hz) | reduziert Autokorrelation + Rechenkosten; Reha-Bewegungen langsam; identisch für alle Modelle |
| Inference separat gebenchmarkt | isoliert Latenz von Data-Loading/Metrik-Berechnung; 50-Frame-Warm-up gegen Init-Artefakte; sequenziell gegen Resource Contention; native Runtimes = Deployment-Realismus |
| Median pro Sequenz (nicht Mittel) | robust gegen Restausreißer innerhalb einer Sequenz |
| COCO nur deskriptiv | anderes Matching + Visibility-Handling → inferenzielle Vergleiche wären protokoll-konfundiert |

## 4b. Verständnis-Klärungen (Session 7.7.2026)

- **COCO torso-center matching (§3.2):** Mehrere GT-Personen pro Bild → Prediction muss
  einer GT-Person zugeordnet werden, sonst vergleicht man Pose von Person A mit GT von
  Person B. Torso-Center statt Box-IoU (MediaPipe hat keine Boxen) und statt OKS
  (bräuchte COCO-Sigmas; Matching über die Bewertungs-Metrik wäre zirkulär);
  pose-stabil, deterministisch, für alle Modelle gleich.
- **Primaries (§3.3):** Nicht „beste Variante", sondern (a) protokollfähig — nur
  MultiPose liefert Kandidaten + Boxen fürs Multi-Person-Protokoll, SinglePose kann es
  strukturell nicht; (b) deployment-typisch — MP Full = API-Default (complexity=1),
  YOLOv8n = Mobile-Einstieg. **Intra-Familien-Analyse = Versicherung: wäre eine andere
  Variante dominant, würde man es sehen — nichts versteckt.**
- **Zwei Schwellen (§3.3/3.9) — Türsteher vs. Qualitätskontrolle:** MP
  detection-confidence 0.1 (API-Default wäre 0.5!) = Detection-Stage IM Modell („ist da
  eine Person?") — gesenkt, weil 0.5 vermeidbare Detection-Failures erzeugte; hilft nur
  ANZUTRETEN, nicht besser auszusehen. Joint-Filter 0.5 = UNSERE Evaluations-Schwelle
  pro Gelenk, identisch für alle Modelle.
- **Keypoint-Mapping (§3.4) — Herkunft:** (1) publizierte Index-Layouts (COCO-17,
  MP-33), (2) joint_names.txt der GT, (3) **MoCap-Namenskonvention: Joint = Name des
  Segments, das an ihm BEGINNT** (LeftArm=Schulter, LeftForeArm=Ellbogen,
  LeftFoot=Knöchel!), (4) visuelle Validierung per Overlay (validate_mapping.py,
  validation_skeleton.png).
- **NMPJPE-Tiefe (§3.5):** Formel in Worten + jede Komponente begründen + Grenzfall
  (N<12 nach Filter → Mittel über valide). Erzählung NICHT „nur übernommen", sondern
  „Auswahl aus 4 gängigen Metriken mit Gründen gegen jede Alternative" (PCK
  verschluckt Fehlergröße, OKS braucht COCO-Sigmas/Objektfläche, Pixel-MPJPE nicht
  vergleichbar). Standard + passend = vollwertige Rechtfertigung + Literatur-
  Vergleichbarkeit.
- **Rotation (§3.6) verstehen:** Schulter-Vektor in 3D-Horizontalebene; frontal →
  Vektor quer (Δz≈0, θ≈0°), seitlich → Vektor in die Tiefe (θ→90°); atan2 mit
  Beträgen faltet auf [0°,90°]. Aus GT statt Prediction = keine Zirkularität
  (Modellfehler würden sonst Bin-Zuordnung verfälschen). Offset 65°: Winkel zwischen
  MoCap-Achse und Kamera-Blickrichtung, empirisch aus 3 frontalen Referenz-Recordings
  (keine Extrinsics publiziert).
- **Augenmaß-Rechtfertigung (3-Schritte-Muster sauberer Messmethodik):** (1)
  Fehlerquelle benannt + quantifiziert (±5°), (2) Analyse-Auflösung angepasst
  (10°-Bins > ±5°; feiner = Scheinpräzision, bimodale Verteilung ließe Mittel-Bins
  leer), (3) Schlüsse skaliert: Offset ist GLOBAL → verschiebt alle Modelle identisch
  → Modell-VERGLEICH invariant; nur absolute Bin-Grenzen tragen Unsicherheit →
  „viewpoint sensitivity under fixed camera geometry, not a dense rotation study".
  Hält vor Gall, wenn proaktiv präsentiert.
- **Person-Selection (§3.7) Korrektheits-Check:** Pseudo-Box aus Keypoint-Extremen
  würde bei Arm-Übungen mit der Übungsphase PULSIEREN → Auswahl könnte mitten in der
  Repetition zum Coach springen; Torso-Extent ist pose-invariant(er). Ehrlich bleiben:
  Heuristik nicht perfekt — die Arbeit zeigt selbst den Fall, wo largest-area den
  Therapeuten wählt (Kap. 6, Panel c) → Selection = Pipeline-Design, separat
  analysiert, nicht als optimal verkauft.
- **Displacement-Details (§3.8):** Nenner = Mittel beider GT-Torso-Längen (Symmetrie —
  der Übergang gehört zu beiden Frames — + glättet Projektions-Schwankungen). Joint
  nur wenn in BEIDEN Frames valid (Sprung braucht zwei Punkte; Auftauchen/Verschwinden
  gehört zu Detection Completeness, nicht Stability — ein Fehlermechanismus pro
  Metrik). Gängigkeit: Konzept ja (SmoothNet/Acceleration-Metriken); unsere Form =
  bewusst einfache, transparente Proxy-Variante.
- **§3.9 akademisch korrekt, Kriterium:** transparent definiert + konsistent
  angewandt + nichts verschwindet unberichtet. Outlier-Schwelle 100% Torso ist
  physikalisch motiviert (Fehler > ganzer Torso = falsche Person/Totalausfall =
  anderer Mechanismus), Outlier werden UMGEBUCHT in die Failure-Analyse, nicht
  gelöscht.
- **§3.12 Zusatz:** Cluster = eine Person mit ALLEN ihren Sequenzen (via person_id).
  Kette: Median pro Sequenz (robust) → Mittel pro Person → 10 Zahlen → exakter Test.
  Video-Level-Sensitivity: 2¹²⁶ Welten nicht enumerierbar → Monte-Carlo-Stichprobe
  (Cluster-Ebene exakt, weil 1024 machbar).

## 4c. Mündliche One-Liner (gedrillt 7.7.2026)

1. **Zwei Schwellen:** "The 0.1 is MediaPipe's internal detector threshold — it only
   decides whether a person is reported at all. The 0.5 is our evaluation filter that
   drops individual low-confidence joints, applied identically to all models."
   (⚠️ 0.1 = gesenkte EINTRITTSHÜRDE, verhindert Detection-Failures — filtert nichts;
   Detection-Confidence, nicht „frame confidence".)
2. **Rotation:** "The calibration offset is global, so any error shifts all models
   identically — the 10-degree bins absorb the ±5-degree uncertainty, and the
   between-model comparison stays valid."
3. **Aggregation:** "Per sequence I take the median frame error for robustness, then
   average those medians per person — that gives ten numbers per model, and the paired
   tests run on those." (⚠️ Median gegen AUSREISSER, nicht „jitter" sagen; „mean",
   nicht „middle".)

## 5. Offene Checks (vor der Verteidigung klären!)

- [x] **Warum 63 von 65 Recordings? GELÖST (9.7.2026):** Die 65 publizierten
  Recordings enthalten eine in zwei Teile gesplittete Ex5-Session (**PM_117a +
  PM_117b**, a/b-Suffixe in allen Dateien) außerhalb des Standard-Namensschemas.
  Evaluiert: die 63 regulären IDs mit beiden Views + GT. EN-Antwort: "The released 65
  include one session split into two part-recordings — PM_117a and PM_117b — outside
  the standard identifier scheme. The evaluation covers the 63 regular identifiers,
  each with both camera views and matching ground truth."
- [ ] **Zusammensetzung der 367.200 model-frame observations** (Frames × Modelle?
  welche Modellmenge?) — in `evaluation_v2/results/` verifizieren.
- [ ] NMPJPE-Definitions-Check: Steht in Section 3.5 ein expliziter Hinweis zur
  Abgrenzung vom 3D-N-MPJPE? (Nein — deshalb Zone-1-Antwort drillen.)

## 6. Folien-Kandidaten aus Kapitel 3

- **Folie "Benchmark":** REHAB24-6 in 1 Bild (Beispiel-Frame mit GT-Skelett) + 5
  Kennzahlen (10 Subjects, 6 Übungen, 2 Views, MoCap-GT, 126 Sequenzen).
- **Folie "Evaluation protocol":** Pipeline-Grafik: Video → frame-independent inference
  → 12-Joint-Mapping → Confidence-Filter → NMPJPE/Displacement → Cluster-Statistik.
  **Pflichtfolie — hier zeigst Du die versteckte Komplexität.**
- **Folie "Six dimensions":** die 6 Evaluationsdimensionen als Icons/Grid.
- Statistik (Cluster + Permutation + Holm) als **halbe Folie oder Backup-Folie** —
  im Vortrag 2 Sätze, Details für die Fragerunde bereithalten.

## 7. Verständnis-Klärungen: Statistik-Fundament (Session 6.7.2026)

### Schritt 0 — Das Grundproblem: Abhängigkeit
367.200 Frames ≠ 367.200 unabhängige Beobachtungen: Frames desselben Videos sind fast
identisch, Videos derselben Person teilen deren Körperbau. Frame-Level-Tests täuschen
riesige Stichproben vor → wertlos kleine p-Werte. **Lösung: Aggregation auf die
natürliche Unabhängigkeitseinheit = Person** (Median pro Sequenz → Mittel pro
Subject-Cluster, N=10 ehrliche Beobachtungen pro Modellpaar).

### Schritt 1 — Paired (gepaart)
Beide Modelle sahen dieselben 10 Personen auf denselben Frames → pro Person Differenz
dᵢ = NMPJPE_A(i) − NMPJPE_B(i). Personen-Eigenschaften (Körperbau, Ausführung) kürzen
sich raus → nur Modellunterschied + Rauschen bleibt. Paarung = massiv mehr Power bei
gleichem N.

### Schritt 2 — Sign-flip-Permutationstest (exakt)
Unter H₀ (Modelle gleich gut) ist das Vorzeichen jeder Differenz ein Münzwurf →
alle 2¹⁰ = 1024 Vorzeichen-Zuweisungen sind gleich wahrscheinlich.
1. Teststatistik: Mittel der 10 beobachteten Differenzen.
2. Statistik in allen 1024 Vorzeichen-Welten berechnen → Nullverteilung.
3. p-Wert = Anteil der Welten mit |Statistik| ≥ |beobachtet| (zweiseitig).
Intuition: Ein Modell gewinnt alle 10 Cluster → nur 2 von 1024 Welten so extrem →
p = 2/1024 ≈ **0,002** (= minimaler zweiseitiger p-Wert!). Exakte Enumeration → keine
Verteilungsannahme, exakt gültig bei N=10. (t-Test bräuchte Normalität der Differenzen
— bei N=10 nicht prüfbar.)

### Schritt 3 — Holm-Korrektur (multiple Vergleiche)
k Tests bei α=0,05 → Fehlalarm-Chance bis 1−0,95^k (k=6: ~26%). Bonferroni: alle p
gegen α/k — konservativ. **Holm (step-down):** p-Werte aufsteigend sortieren; p₍₁₎
gegen α/k, p₍₂₎ gegen α/(k−1), …, beim ersten Scheitern stoppen. Kontrolliert die
family-wise error rate genauso streng wie Bonferroni, ist aber **uniformly more
powerful** — es gibt keinen Grund für Bonferroni.

### Schritt 4 — Percentile-Bootstrap-CI (Effektgröße)
p-Wert sagt „real?", CI sagt „wie groß + wie unsicher". Bootstrap: aus den 10
Cluster-Differenzen 10 Werte **mit Zurücklegen** ziehen („was wäre bei 10 anderen
Personen"), Mittel berechnen, **20.000×** wiederholen → empirische Verteilung;
95%-CI = [2,5%-Quantil, 97,5%-Quantil]. 20.000 Resamples → stabile Rand-Quantile.

### Durchgerechnetes Mini-Beispiel (3 Personen — zum Wiederholen)
Differenzen d = Fehler_A − Fehler_B: Person 1: +2, Person 2: +1, Person 3: +3
(positiv = B besser). Beobachtetes Mittel: +2,00. Unter H₀ ist jedes Vorzeichen ein
Münzwurf → 2³ = 8 gleich wahrscheinliche Welten:

| Vorzeichen | Mittel | | Vorzeichen | Mittel |
|---|---|---|---|---|
| + + + | **+2,00** (beobachtet) | | − + + | +0,67 |
| + + − | 0,00 | | − + − | −1,33 |
| + − + | +1,33 | | − − + | 0,00 |
| + − − | −0,67 | | − − − | **−2,00** |

p = Anteil Welten mit |Mittel| ≥ 2,00 = 2/8 = **0,25** → bei N=3 nie signifikant,
selbst wenn B überall gewinnt. Bei N=10: 2/1024 ≈ 0,002. **Vorzeichen flippen = bei
dieser Person die Ergebnisse von A und B vertauschen.** Erlaubt, weil unter H₀
(„kein echter Unterschied") die Differenz pures Rauschen ist — und Rauschen hat keine
Lieblingsrichtung (50/50). Größen bleiben unverändert: Sie SIND die echten Beispiele
der Rauschgröße; H₀ garantiert nur die Zufälligkeit der Richtung.

**p-Wert-Richtung:** p = Plausibilität der Zufalls-Erklärung. Klein = Zufall
unplausibel = signifikant. ⚠️ p ist NICHT „Wahrscheinlichkeit, dass H₀ stimmt",
sondern „Wahrscheinlichkeit eines so extremen Ergebnisses, ANGENOMMEN H₀ stimmt".

### Holm-Zahlenbeispiel (zum Wiederholen)
3 Paar-Tests, p = 0,002 / 0,020 / 0,040, α = 0,05:
- **Bonferroni:** alle gegen 0,05/3 ≈ 0,0167 → nur 0,002 signifikant.
- **Holm:** 0,002 < 0,0167 ✓ → 0,020 < 0,05/2 = 0,025 ✓ → 0,040 < 0,05/1 ✓ →
  **alle drei signifikant.** Gleiche Fehlalarm-Garantie (family-wise error rate),
  mehr Power. Ohne Korrektur wäre bei k Tests die Fehlalarm-Chance bis 1−0,95^k
  (k=3: ~14%).

### Bootstrap-Intuition (zum Wiederholen)
Ideal-Experiment (unmöglich): Studie hundertfach mit je 10 neuen Personen wiederholen
→ Streuung der Mittelwerte. Trick: neue 10er-Gruppen aus den eigenen 10 Differenzen
**mit Zurücklegen** ziehen (ohne Zurücklegen: immer dieselbe Gruppe, null Variation).
20.000 Pseudo-Gruppen → 20.000 Mittelwerte → mittlere 95% = Konfidenzintervall.
CI schließt 0 nicht ein ⇔ konsistent mit signifikantem Test.

### Simpel-Versionen (Merkbilder aus dem Drill, 7.7.2026)
- **Vorzeichen-Flip = „Was wäre, wenn bei dieser Person das andere Modell gewonnen
  hätte?"** (Person 2: A=12/B=11 → geflippt A=11/B=12). Erlaubt, weil H₀ = „zwei exakt
  gleich starke Tischtennisspieler": Wer einen Satz gewinnt, ist Münzwurf → beide
  Richtungen 50/50. Einer gewinnt 10 von 10 Sätzen → niemand glaubt mehr an „gleich
  stark" (p = 2/1024).
- **p-Wert prüfungssicher:** „WENN in Wahrheit kein Unterschied bestünde, würde reiner
  Zufall ein so extremes Ergebnis nur in p·100 von 100 Fällen erzeugen." NICHT: „zu p%
  ist es Zufall". p=0,03 < 0,05 → signifikant.
- **Holm-Lotterie:** Schutzfaktor = Anzahl möglicher Fehlalarm-Lose. Anfangs 3 Lose →
  Hürde α/3. Kleinster p-Wert nimmt die härteste Hürde → Effekt gilt als echt → nur
  noch 2 mögliche Zufalls-Lose → α/2 reicht → dann α/1. Sortierung: Wenn der stärkste
  Kandidat die härteste Hürde nicht schafft, müssen die schwächeren nicht mehr antreten.

### Rollenverteilung (Q&A-Framing)
> *Permutation test: is the difference real? Bootstrap interval: how large, with what
> uncertainty? Holm: how do we stay honest when asking several questions at once?*

EN-Musterantworten: `qa_catalog.md` B1–B5.
