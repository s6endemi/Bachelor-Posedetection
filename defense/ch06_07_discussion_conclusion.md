# Chapter 6 (Discussion) + 7 (Conclusion) — Verteidigungsanalyse

> Quelle: `thesis/chapters/06_discussion.tex`, `07_conclusion.tex` (final, vollständig gelesen)
> Lernziel hier ≠ Zahlen (das ist Kap. 5), sondern **Claim-Kalibrierung**: für jede
> Aussage wissen, wie stark sie behauptet wird und warum nicht stärker.
> Leitsatz des Kapitels: "separates direct empirical findings from narrower
> interpretations and clearly marked open questions."

## 1. Die 5 RQ-Antworten (6.1) — frei sprechen können!

- **RQ1 (accuracy+completeness):** MoveNet niedrigster Fehler (frame- UND cluster-level,
  nach Holm vor beiden; MediaPipe vor YOLO). Completeness folgt ANDERER Ordnung
  (YOLO > MP > MoveNet) → Vergleich trennt Accuracy von Completeness, kein Modell
  gewinnt beides.
- **RQ2 (stability):** MoveNet niedrigstes Displacement, dann YOLO, dann MediaPipe;
  alle 3 MoveNet-Varianten im niedrigsten Bereich → verstärkt das Muster, kein
  separater Trade-off.
- **RQ3 (robustness):** **antwortabhängig von der Bedingung** — Viewpoint: MoveNet am
  robustesten; Segmentation-Multi-Person: keiner degradiert stark; Coach-Extremfall:
  MediaPipe niedrigste Katastrophen-Rate. Drei verschiedene Robustheits-Regime, nicht
  zu einem Skalar kollabierbar.
- **RQ4 (transfer):** "limited rather than absent" — Nine-Model-Leader wechselt
  (auch bei Single-Person-COCO); Hauptmodelle: MoveNet auf beiden #1, MP/YOLO
  tauschen. Protokoll-gebunden → bounded transferability, kein protokoll-
  unabhängiger Ranking-Claim.
- **RQ5 (recommendations):** use-case-abhängig; Default für Body-Joint+CPU: MoveNet
  MultiPose (Accuracy + Stabilität + Throughput); YOLO wenn Completeness dominiert;
  MediaPipe wenn reichere Landmarks / einfachere Integration / Coach-Robustheit
  zählen.

## 2. Evidenz-Level-Landkarte (DAS Prüfungskonzept)

| Level | Aussagen | Formulierung im Text |
|---|---|---|
| **Fakt** | MoveNet bester Fehler/Displacement/Throughput; YOLO Completeness; MP Coach-Rate; 16/23/62; Hip-Gap 2,01→1,21pp | direkte Zahlen |
| **Gestützte Interpretation** | „Detection count alone does not explain coach robustness" (3 Paarvergleiche!); limited transferability; Hip-Fehler konzentriert, nicht uniform | "indicates", "consistent with", "supports" |
| **Offene Hypothese** | Failure-Signaturen ↔ Paradigmen; Hip = Landmark-Definitions-Mismatch; MoveNet-Coach-Mechanismus | "observational rather than mechanistic", "plausible hypothesis for future work", "cannot be localized further", "evidence remains indirect" |

**Regel: Immer auf dem Evidenz-Level der Arbeit antworten.** Bei Kausal-Suggestionen
(„so architecture causes X?") kontrolliert zurückstufen.

## 3. Die drei Profile (6.2) — Ein-Satz-Steckbriefe

- **YOLOv8 — high-sensitivity one-stage:** detektiert am vollständigsten (79%) UND
  exponiert am meisten Zweitpersonen (62%) → am häufigsten zur Disambiguierung
  gezwungen; schwächer bei Accuracy + Viewpoint.
- **MoveNet — low-exposure bottom-up:** beste Accuracy + Stabilität + Throughput;
  konservativste Exposure (16%) — die sich aber NICHT in niedrigster Coach-Rate
  niederschlägt (der offene Punkt!).
- **MediaPipe — filtered top-down:** mittlere Exposure (23%), niedrigste
  Katastrophen-Rate im Coach-Fall; langsamer + ungenauer auf den 12 Joints; eigenes
  Deployment-Profil, kein Gesamtsieger.
- **Failure-Signature (Hypothese!):** top-down commit-then-predict → Keypoint-
  Displacement; bottom-up per-Joint-Confidence → Confidence-Collapse (graceful
  degradation); one-stage Kopplung → Missing-Detection. Confound selbst benannt:
  **1 Modell pro Paradigma** → Architektur nicht von Trainingsdaten/Kalibrierung/
  Landmark-Definitionen trennbar.

## 4. Cross-Profile-Argument (6.3) — die 3-Schritte-Erzählung

1. **MP–YOLO (konsistent):** 62% vs. 23% Exposure → mehr Gelegenheiten für
   Misassignment; Fig. 5.4 konsistent damit.
2. **MP–MoveNet (bricht die Erklärung):** MoveNet exponiert WENIGER (16% < 23%),
   failt aber MEHR (13,8% > 9,1%) → Detection-Counts erklären es nicht; Rest-
   Mechanismus in candidate retention / thresholding / post-detection selection;
   Panel (c): Detection ok, largest-area wählt Therapeuten = Selektions-Stufe.
   Framework sieht Interna nicht → nicht weiter lokalisierbar.
3. **MoveNet–YOLO (bestätigt):** 4× Exposure-Unterschied (16 vs. 62), fast identische
   Outlier-Raten (13,8 vs. 13,9) → keine monotone Beziehung.
→ Zusammen: "the strongest mechanistic conclusion the data support" — bewusst eine
NEGATIVE Konklusion (was es NICHT ist). Stärke, nicht Schwäche.

## 5. Hip-Interpretation (6.4)

Fakt: MP-Hips 16,72% vs. MoveNet 11,28%; ohne Hips Gap 2,01→1,21pp (≈40% des Gaps).
Hypothese: **Landmark-Definitions-Mismatch** (BlazePose-Körpermodell verortet Hüfte
anders als MoCap-Skelett) — gestützt durch KONZENTRATION des Fehlers (nicht uniform),
explizit "indirect". ⚠️ Abgrenzung für Q&A: Das MAPPING (Index↔GT-Joint) ist visuell
validiert; die Hypothese betrifft die anatomische DEFINITION des Landmarks.

## 6. Empfehlungen (6.7) + der Bonus-Satz

Default: MoveNet MultiPose (Body-Joint + CPU). Speed → SP_Lightning (100 FPS, nur
wenn Accuracy-Verlust ok). Completeness → YOLO (gegen schwächere Viewpoint-Robustheit
+ höhere Coach-Rate abwägen). Reichere Landmarks/Integration/Coach → MediaPipe.
**Bonus-Satz (RQ5-Abbinder):** "Deployment decisions can matter as much as the model
choice itself" — frontale Kamera-Anleitung, Interferenz-Monitoring, Qualitätswarnungen.

## 7. Limitations (6.8) — als Scope-Entscheidungen + Future-Work-Anschluss

| Limitation | Framing | Future-Work-Anschluss |
|---|---|---|
| Healthy volunteers, kein Patient-Kohort | zentrale Limitation, Proxy-Framing von Anfang an | Patient-Validierung = wichtigster nächster Schritt |
| 10 Cluster (person_id) | viel stärker als Frame-Unabhängigkeit, aber klein | größere Kohorten |
| Coach n=5, nur externes Verhalten | reicht zum Unterscheiden der Paar-Muster, nicht zur Mechanismus-Lokalisierung | interne Kandidaten/Thresholds loggen |
| Frame-independent | Design-Entscheidung (Modell vs. Pipeline); Werte = upper bound | uniformes temporales Post-Processing für alle |
| Displacement-Proxy | subtrahiert GT-Bewegung nicht; Vergleich fair unter shared protocol | — |
| COCO-Protokoll-Differenzen | limited transferability, kein sauberer Domain-Shift-Schätzer | standardisiertes Cross-Dataset-Protokoll |
| CPU-only | relative Rankings nutzbar; Mobile-HW kann Ordnung ändern | Smartphone-Benchmark mit Delegates |
| 2D-only | wichtige Vorstufe, nicht das volle Assessment-Problem | 3D-Winkel/Depth |

**Drill-Format:** Limitation genannt bekommen → (a) warum bewusste Entscheidung,
(b) welcher Future-Work-Punkt sie adressiert. Nie eine Schwäche ohne nächsten Schritt.

## 8. Kapitel 7 (Conclusion) — die Schlussfolien-Vorlage

- **Zwei Kernbefunde:** (1) MoveNet MultiPose = stärkster Default (niedrigster
  Cluster-Fehler + niedrigstes Displacement). (2) Drei architektonische Profile statt
  eines Rankings; Exposure erklärt Coach-Robustheit nicht; Failure-Decomposition
  verstärkt das (3 distinkte Signaturen).
- **Drei Lese-Limits:** healthy volunteers (kein Patient-Kohort); COCO = anderes
  Protokoll (limited transferability, nicht Benchmark-Invalidität); frame-independent
  (unsmoothed outputs, kein fertiges Pipeline-Verhalten).
- **Schluss:** Default-Empfehlung + wann welche Alternative + Patient-Validierung als
  Weg zur Klinik. → Der Conclusion-Text IST die Vorlage für die letzten 2 Folien.

## 8b. Mündliches Skript Kapitel 6 (EN, pro Subchapter)

- **6.1:** "Each research question gets a direct answer — and the honest theme is that
  the answers are dimension-dependent: accuracy, completeness, and robustness order
  the models differently."
- **6.2:** "The models form three consistent characters: YOLOv8 sees the most — and
  must disambiguate the most; MoveNet is the conservative all-rounder — yet
  conservatism doesn't buy coach safety; MediaPipe is the filtered profile with the
  lowest catastrophic rate. Each fails in its own architectural style — reported as an
  observational correspondence, not causation."
- **6.3:** "Three pairwise comparisons tell one story: MediaPipe–YOLO fits the
  detection-count explanation, MediaPipe–MoveNet breaks it, MoveNet–YOLO confirms the
  break. Detection count alone does not explain coach robustness — the residual
  mechanism sits at the selection stage and is deliberately left open."
- **6.4:** "MediaPipe's error is concentrated in the hips; removing them closes 40% of
  the gap — consistent with a landmark-definition mismatch rather than uniform
  degradation. Indirect evidence, marked as such."
- **6.5:** "The ranking shift is real and persists on single-person images — but the
  comparison is protocol-bound, so it supports limited transferability, not benchmark
  invalidity."
- **6.6:** "Closest neighbor is the REHAB24-6 paper itself — I add MoveNet, temporal
  displacement, and the coach analysis; against broader clinical comparisons my work
  is narrower in models but deeper in deployment characterization."
- **6.7:** "Default: MoveNet MultiPose. YOLOv8 when completeness dominates, MediaPipe
  for coach robustness or richer landmarks, SP Lightning when only speed matters — and
  deployment guidance can matter as much as the model choice."
- **6.8/6.9:** "Every limitation is a scope decision with a named next step — the most
  important one being validation on real patient data."

## 9. Gefährliche Fragen (EN-Kernantworten)

**"One model per paradigm — how do you know it's the architecture?"**
> "I don't claim causation — the thesis marks the correspondence as observational.
> With one family per paradigm, architecture is confounded with training data,
> confidence calibration, and landmark definitions. What I report is that each model's
> dominant failure category is *consistent with* its design logic, as a hypothesis for
> future work."

**"Maybe your hip mapping is just wrong?"**
> "The index mapping is validated visually. The hypothesis concerns the landmark
> *definition* — where BlazePose's body model anatomically places the hip versus the
> MoCap skeleton. The error is concentrated in one structurally specific region and
> removing the hips closes about 40% of the gap — consistent with a definition
> mismatch, and the thesis marks the evidence as indirect."

**"Why trust REHAB rankings for patients if they don't even transfer to COCO?"**
> "That's exactly why patient validation is the first item of future work. But
> structurally, REHAB24-6 is far closer to the deployment setting — guided exercises,
> fixed cameras, a visible therapist — so the remaining transfer gap is much smaller
> than to COCO, which differs in both domain and protocol."

**"Five coach videos — enough?"**
> "It's a named limitation. The coach analysis makes qualitative distinctions — three
> different pairwise patterns and one counter-example video where all three models
> fail — and I report it descriptively, without inferential claims on that subset."

**"Why not just crown MoveNet?"**
> "Because deployment is multi-criteria. If your downstream processing needs all
> twelve joints, YOLOv8 delivers them most often; if coach-scene robustness or the
> richer 33-landmark output matters, MediaPipe is preferable. Three profiles is the
> decision-relevant answer, not diplomacy."

## 10. Folien-Kandidaten

- **RQ-Antworten-Folie(n):** 5 RQs mit Ein-Zeilen-Antworten (Spiegelung der
  RQ-Folie vom Anfang — schließt den Kreis).
- **Drei-Profile-Folie:** DIE Kernfolie des Vortrags (je Modell 2 Zeilen Profil).
- **„Detection count ≠ robustness"-Folie:** die 3 Paarvergleiche als Mini-Grafik
  (16/23/62 vs. 9,1/13,8/13,9).
- **Limitations-Folie:** 3–4 wichtigste (Population, frame-independent, Protokoll,
  2D) als bewusste Scope-Grenzen + Future-Work-Pfeile.
- **Schlussfolie:** Conclusion-Absatz 2+4 kondensiert (2 Befunde + Empfehlung).
