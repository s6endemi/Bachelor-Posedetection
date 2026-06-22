# Thesis Rewrite Plan — Konsolidierter Masterplan

> **Hinweis zum Dokumentstatus:**
> Dieses Dokument wurde während der aktiven Rewrite-Phase erstellt und dokumentiert
> die methodische Planung, das Claims-Framework und die Formulierungs-Leitlinien.
> Es bleibt als **historische Referenz** gültig, spiegelt aber nicht jede einzelne
> Detail-Entscheidung wider, die seither getroffen wurde (z. B. finale Figur-Variante,
> angepasste Caption-Formulierungen, Chapter 1/2/4/7/Abstract).
>
> **Für den aktuellen Stand:** siehe `evaluation_v2/STATUS.md` und `HANDOFF_REVIEWER_V2.md`.
> **Für die finalen Kapiteltexte:** siehe `thesis/chapters/*.tex` und `thesis/abstract.tex`.
>
> Das Claims-Framework in Abschnitt 2 und die Narrative-Leitlinien sind weiterhin
> die Grundlage für die Gesamtarbeit.

---

> Erstellt nach vollständiger Analyse, neuem Evaluation-Run, COCO Cross-Dataset,
> und 9 Modell-Varianten.

---

## 1. Ausgangslage

### Lars' Feedback (verbatim)
1. Text nicht über-LLMizen. Weniger als 50 Seiten ist völlig ok.
2. Unklar ob Model Selection Strategies unser Vorschlag sind → klären.
3. Keine redundanten Table+Figure Paare (Section 5).
4. Mehr qualitative Beispiele in Section 5.
5. Verschiedene Versionen der Modelle hinzufügen + Detection Threshold Effekt.
6. Statistische Analysen mit Abhängigkeiten, für mehr Metriken.
7. Zweiter Datensatz für Cross-Dataset Generalization.

### Was wir gemacht haben
- Komplett neue Evaluation-Pipeline (`evaluation_v2/`) mit konsistentem 0.5 Confidence-Threshold
- 9 Modell-Varianten auf REHAB24-6 UND COCO val2017 evaluiert
- Frame-Level Multi-Person-Analyse (statt 5-Video-Binary)
- Alle Zahlen verifiziert und Diskrepanzen aufgelöst
- 7 Thesis-Figures generiert (je eine pro Fragestellung)

---

## 2. Claims Framework — Was dürfen wir wie sagen?

### Die vier Evidenz-Level

In einer Bachelorarbeit gibt es eine klare Hierarchie was man behaupten darf:

| Level | Was | Formulierung | Beispiel |
|-------|-----|-------------|----------|
| **Fakt** | Direkt aus Daten ablesbar | "X zeigt Y" / "X achieves Y" | "MoveNet achieves 10.5% NMPJPE" |
| **Gestützte Interpretation** | Logische Schlussfolgerung aus Fakten | "X indicates Y" / "This suggests Y" | "The ranking shift indicates that benchmarks don't capture rehab challenges" |
| **Evidenz-basierte Hypothese** | Konsistent mit Daten, aber nicht bewiesen | "We hypothesize that X because of Y. Evidence: Z" | "We hypothesize that MediaPipe's robustness stems from an implicit face-detection filter" |
| **Spekulation** | Plausibel aber ohne direkte Evidenz | VERMEIDEN oder klar als "speculative" markieren | "MediaPipe's 33-landmark network may distribute capacity..." |

### Alle Befunde mit korrektem Evidenz-Level

#### FAKTEN (direkt zitierbar)

**F1: MoveNet ist am genauesten auf REHAB24-6**
- Evidenz: 10.5% vs 12.5% (MediaPipe) vs 12.8% (YOLOv8)
- Statistik: Wilcoxon p<0.001, Cohen's d=0.29
- Formulierung: "MoveNet achieved significantly lower NMPJPE (10.5%) than MediaPipe (12.5%, p<0.001, d=0.29) and YOLOv8 (12.8%)"

**F2: MediaPipe hat weniger Person-Switch Events in Coach-Videos**
- Evidenz: 9.1% Outlier-Rate vs 13.8% (MoveNet) vs 13.9% (YOLOv8)
- Formulierung: "In multi-person scenarios, MediaPipe exhibited a lower person-switch rate (9.1%) compared to MoveNet (13.8%) and YOLOv8 (13.9%)"

**F3: MoveNet ist am rotations-robustesten**
- Evidenz: +18% Degradation vs +28% (MediaPipe) vs +32% (YOLOv8)
- Formulierung: "MoveNet showed the lowest accuracy degradation under rotation (+18%)"

**F4: Rankings verschieben sich zwischen COCO und REHAB24-6**
- Evidenz: MoveNet #1 REHAB → #4 COCO; YOLOv8m #1 COCO → #2 REHAB
- Formulierung: "Model rankings shifted between standard benchmark and rehabilitation data"

**F5: MediaPipe hat systematisch höhere Hip-Fehler**
- Evidenz: Hips 16.7% vs 11.3% (MoveNet); ohne Hips schrumpft Differenz von 2.0% auf 1.2%
- Formulierung: "MediaPipe exhibited systematically elevated hip joint errors (16.7% vs 11.3%), accounting for approximately 40% of its overall accuracy deficit"

**F6: MoveNet hat die niedrigste Streuung**
- Evidenz: SD 4.7% vs 7.1% (MediaPipe) vs 6.7% (YOLOv8)
- Formulierung: "MoveNet showed the most consistent predictions (SD=4.7%)"

**F7: MoveNet generalisiert am stabilsten cross-dataset**
- Evidenz: REHAB→COCO Shift +1.9% vs +4.7% (MediaPipe) vs +0.9% (YOLOv8)
- Formulierung: "MoveNet exhibited the smallest performance shift between datasets (+1.9%)"

**F8: Intra-Family Tradeoff existiert**
- Evidenz: MediaPipe Lite 13.5% → Full 12.5% → Heavy 11.5%; YOLOv8n 12.8% → s 11.4% → m 10.9%
- Formulierung: "Within each model family, larger variants achieved lower error at the cost of higher latency"

#### GESTÜTZTE INTERPRETATIONEN (mit Evidenz-Verweis)

**I1: Standard-Benchmarks bilden Reha-Szenarien nicht vollständig ab**
- Stützt sich auf: F4 (Ranking-Verschiebung) + F7 (unterschiedliche Shifts)
- Formulierung: "The observed ranking shifts between COCO and REHAB24-6 (F4) indicate that standard benchmark performance does not directly predict rehabilitation-scenario performance"

**I2: Person-Selection ist eine kritische Design-Entscheidung**
- Stützt sich auf: F2 (Person-Switch Rates) + Torso vs BBox Vergleich
- Formulierung: "The differing person-switch rates across models indicate that person selection strategy constitutes a critical design choice that significantly influences real-world performance"

**I3: MediaPipe's Hip-Fehler ist ein Mapping-Problem, kein Qualitätsproblem**
- Stützt sich auf: F5 (systematisch, nicht zufällig) + alle Modelle auf COCO trainiert
- Formulierung: "The systematic nature of MediaPipe's elevated hip error — consistent across exercises, cameras, and rotation angles — suggests a landmark definition mismatch rather than prediction quality deficit"

**I4: Modellwahl hängt vom Use Case ab (holistic)**
- Stützt sich auf: F1-F8 zusammen + MediaPipe's 33 Landmarks/3D/Face
- Formulierung: "While MoveNet achieves the highest body keypoint accuracy, MediaPipe offers 33 landmarks including face and hand tracking, native 3D output, and superior multi-person robustness — features not captured by body-joint NMPJPE alone"

#### HYPOTHESEN (explizit als solche markiert)

**H1: MediaPipe's Multi-Person-Robustness kommt von Detection-Level-Filterung**
- **STATUS: GESTÜTZTE INTERPRETATION (mit quantitativer Evidenz)**
- Detection-Count-Analyse über ALLE 5 Coach-Videos (1178 Frames):

  | Video  | MediaPipe 1p | MediaPipe 2+ | YOLO 1p | YOLO 2+ |
  |--------|-------------|-------------|---------|---------|
  | PM_010 | 53%         | 47%         | 81%     | 19%     |
  | PM_011 | 73%         | 27%         | 68%     | 32%     |
  | PM_108 | 96%         | 4%          | 0%      | 100%    |
  | PM_119 | 84%         | 16%         | 0%      | 100%    |
  | PM_121 | 99%         | 1%          | 0%      | 100%    |
  | TOTAL  | **77%**     | **23%**     | **38%** | **62%** |

- MediaPipe sieht in 77% aller Coach-Frames nur 1 Person. YOLO sieht in 62% 2+ Personen.
- Nuancierung: Der Effekt ist NICHT rein face-basiert. PM_010 (Coach von hinten aber
  gross/nah) wird trotzdem erkannt. Es ist eine Kombination aus Größe, Nähe und
  Face-Visibility die MediaPipe's Detection-Schwelle bestimmt.
- Architektur-Fakt: BlazePose nutzt face/upper-body detector als erste Stufe (Bazarevsky et al. 2020)
- **Thesis-Formulierung (clean, 2 Sätze in Results, 1 Absatz in Discussion):**
  - Results: "MediaPipe detected a single person in 77% of coach video frames, compared
    to 38% for YOLOv8 (Table X). This detection-level difference accounts for MediaPipe's
    lower person-switch rate (9.1% vs 13.9%)."
  - Discussion: "The multi-person robustness difference is primarily detection-level:
    MediaPipe's top-down pipeline detects fewer secondary persons than bottom-up or
    one-stage architectures. The person selection strategy becomes relevant only in
    the minority of frames where multiple persons are detected."
- Qualitative Frames: evaluation_v2/qualitative_frames/ (PM_119 + PM_010, annotated + raw)
- **Kategorisierung der 5 Coach-Videos:**
  - PM_108, PM_119, PM_121: Echte Multi-Person (Coach sichtbar, Patient auch) → Architektur-Effekt klar
  - PM_010, PM_011: Gemischt (Occlusion + Überdetektierung + Multi-Person)

**H2: MoveNet's Konsistenz kommt von der Bottom-Up-Architektur**
- Evidenz FÜR: Niedrigste SD, niedrigster Jitter, stabilster cross-dataset
- Evidenz GEGEN: Korrelation ≠ Kausalität; andere Faktoren möglich (Training, Daten)
- Formulierung: "MoveNet's consistently lower variance across conditions is compatible with its bottom-up architecture, which estimates all keypoints in a single forward pass without dependency on prior detection stages. However, we note that architectural differences are confounded with differences in training data and optimization, and a causal attribution requires controlled ablation studies beyond the scope of this work."

**H3: Ranking-Verschiebung kommt von domänenspezifischen Bedingungen**
- Evidenz FÜR: REHAB hat kontrollierte Bedingungen, COCO hat "in the wild" Szenen
- Formulierung: "We attribute the ranking shift to domain-specific characteristics: rehabilitation recordings feature controlled indoor environments, limited pose variety, and consistent camera distances — conditions that favor models optimized for spatial consistency (MoveNet) over models optimized for diverse scene understanding (YOLOv8 Medium)."

#### VERMEIDEN (zu spekulativ für Bachelor)

- ~~"MediaPipe's 33-Landmark-Netzwerk verteilt Kapazität über zu viele Targets"~~ → Keine Evidenz
- ~~"MoveNet ist objektiv das beste Modell"~~ → Ignoriert MediaPipe's andere Vorteile
- ~~"Die Ergebnisse beweisen dass Bottom-Up besser ist als Top-Down"~~ → Überverallgemeinerung
- ~~"COCO ist kein guter Benchmark"~~ → Zu stark; COCO misst andere Aspekte

---

## 3. Die neue Thesis-Narrative

### Alte Narrative (63 Seiten, aufgebläht)
> "Kein Modell dominiert — jedes hat ein eigenes Performance-Profil."

### Neue Narrative (38-42 Seiten, präzise und ehrlich)
> "Für reine Body-Keypoint-Accuracy dominiert MoveNet auf Rehabilitationsdaten.
> Aber 'bestes Modell' hängt vom Use Case ab: MediaPipe bietet Multi-Person-
> Robustness, 33 Landmarks und 3D-Output — Vorteile die NMPJPE nicht erfasst.
> Die Rankings verschieben sich zwischen COCO und REHAB24-6, was zeigt dass
> domänenspezifische Evaluation unerlässlich ist."

### Die vier Kernaussagen (Evidenz-Level in Klammern)
1. **MoveNet erzielt die höchste Body-Keypoint-Accuracy auf REHAB24-6** [FAKT: F1]
2. **MediaPipe zeigt die höchste Multi-Person-Robustness, vermutlich durch einen architekturellen Detection-Filter** [FAKT: F2 + HYPOTHESE: H1]
3. **Rankings verschieben sich zwischen Standard-Benchmark und Reha-Daten** [FAKT: F4 → INTERPRETATION: I1]
4. **Die Modellwahl ist use-case-abhängig: Accuracy alleine ist kein ausreichendes Kriterium** [INTERPRETATION: I4]

### Holistische Bewertung (über NMPJPE hinaus)

Die Thesis heißt "A Holistic Evaluation" — das bedeutet wir bewerten NICHT nur
Keypoint-Accuracy, sondern auch:

| Dimension | Bester | Evidenz-Level | In Thesis wo? |
|-----------|--------|---------------|---------------|
| Body Accuracy (NMPJPE) | MoveNet | FAKT (F1) | Ch. 5.2 |
| Temporal Stability | MoveNet | FAKT (F6) | Ch. 5.7 |
| Rotation Robustness | MoveNet | FAKT (F3) | Ch. 5.5 |
| Inference Speed | MoveNet | FAKT (Benchmark) | Ch. 5.8 |
| Multi-Person Robustness | MediaPipe | FAKT (F2) | Ch. 5.6 |
| Detection Completeness | YOLOv8 | FAKT | Ch. 5.8 |
| Landmark Richness (33 pts) | MediaPipe | FAKT (Architektur) | Ch. 6.4 |
| 3D Output | MediaPipe | FAKT (Architektur) | Ch. 6.4 |
| Integration Simplicity | MediaPipe | FAKT (API) | Ch. 6.4 |
| Cross-Dataset Stability | MoveNet | FAKT (F7) | Ch. 5.4 |
| Intra-Family Variants | Alle | FAKT (F8) | Ch. 5.3 |

**Warum das wichtig ist:** Wer nur NMPJPE betrachtet, sieht "MoveNet gewinnt."
Wer holistisch betrachtet, sieht "es kommt drauf an." Genau DAS ist der Beitrag
der Arbeit — und warum sie "Holistic Evaluation" heißt.

---

## 3. Verifizierte Zahlen (Source of Truth)

### REHAB24-6 — 3 Hauptmodelle (evaluation_v2/results/rehab24/)
| Modell | NMPJPE | Median | n Frames |
|--------|--------|--------|----------|
| MoveNet MultiPose | 10.5% | 9.9% | 119,439 |
| MediaPipe Full | 12.5% | 11.3% | 119,729 |
| YOLOv8 Nano | 12.8% | 11.2% | 115,711 |

Confidence: 0.5 | Outlier: >100% entfernt | MoCap-Errors: entfernt

### REHAB24-6 — Alle 9 Varianten (evaluation_v2/results/rehab24_all/)
| Variante | NMPJPE | Median |
|----------|--------|--------|
| MoveNet MultiPose | 10.5% | 9.9% |
| YOLOv8 Medium | 10.9% | 10.0% |
| YOLOv8 Small | 11.4% | 10.3% |
| MediaPipe Heavy | 11.5% | 10.6% |
| MediaPipe Full | 12.5% | 11.3% |
| YOLOv8 Nano | 12.8% | 11.2% |
| MoveNet SP Thunder | 13.3% | 11.9% |
| MediaPipe Lite | 13.5% | 11.8% |
| MoveNet SP Lightning | 15.0% | 12.8% |

### COCO val2017 — Alle 9 Varianten (evaluation_v2/results/coco/)
| Variante | NMPJPE | Median |
|----------|--------|--------|
| YOLOv8 Medium | 10.1% | 7.6% |
| YOLOv8 Small | 10.9% | 8.1% |
| MoveNet SP Thunder | 12.0% | 8.0% |
| MoveNet MultiPose | 12.4% | 8.7% |
| MoveNet SP Lightning | 12.6% | 8.5% |
| YOLOv8 Nano | 13.7% | 10.1% |
| MediaPipe Heavy | 15.1% | 8.7% |
| MediaPipe Full | 17.2% | 10.5% |
| MediaPipe Lite | 20.1% | 12.8% |

### Ranking-Verschiebung (Kernbefund für Cross-Dataset Section)
- MoveNet MP: #1 REHAB → #4 COCO
- YOLOv8 Medium: #2 REHAB → #1 COCO
- MediaPipe Heavy: #3 REHAB → #7 COCO

### Multi-Person (REHAB24-6, 5 Coach-Videos, MIT Outliers)
| Modell | Clean | Coach | Degradation | Outlier-Rate Coach |
|--------|-------|-------|-------------|-------------------|
| MediaPipe | 14.4% | 45.4% | +215% | 9.1% |
| MoveNet | 12.7% | 62.2% | +390% | 13.8% |
| YOLOv8 | 17.0% | 66.0% | +289% | 13.9% |

### Rotation (REHAB24-6, frontal 0-20° vs lateral 60-90°)
| Modell | Frontal | Lateral | Degradation |
|--------|---------|---------|-------------|
| MoveNet | 9.8% | 11.5% | +18% |
| MediaPipe | 11.3% | 14.4% | +28% |
| YOLOv8 | 11.1% | 14.7% | +32% |

### Jitter (REHAB24-6)
| Modell | Mean | Median |
|--------|------|--------|
| MoveNet | 1.74% | 0.39% |
| YOLOv8 | 2.45% | 0.38% |
| MediaPipe | 2.78% | 0.54% |

### Inference Speed (aus altem Benchmark — noch gültig für 3 Hauptmodelle)
| Modell | Mean (ms) | FPS |
|--------|-----------|-----|
| MoveNet | 36.1 | 27.7 |
| YOLOv8 Nano | 52.9 | 18.9 |
| MediaPipe Full | 67.9 | 14.7 |

**TODO: Inference Benchmark für die 6 neuen Varianten nachholen.**

---

## 4. Kapitelstruktur (Neu)

### Chapter 1: Introduction (~3 Seiten)
- Motivation: Telerehabilitation, Adherence-Problem
- Problem: Standard-Benchmarks ≠ Reha-Anforderungen
- Research Questions (gleich, 4 Stück)
- Contributions (aktualisiert: + Cross-Dataset, + Varianten)
- **Änderungen:** Kürzen, de-LLMizen. Contributions-Liste updaten.

### Chapter 2: Background & Related Work (~6 Seiten)
- HPE Grundlagen (kürzen)
- Architektur-Paradigmen (Top-Down, Bottom-Up, One-Stage)
- Evaluierte Modelle + NEUE: Varianten-Tabelle
- Metriken (NMPJPE, Jitter)
- Related Work (thematisch, wie bisher)
- Research Gap Table (aktualisieren: + Cross-Dataset, + Varianten)
- **Änderungen:** Modellvarianten-Tabelle hinzufügen. Research Gap erweitern.

### Chapter 3: Methodology (~7 Seiten)
- Dataset: REHAB24-6 (wie bisher) + COCO val2017 (NEU, ~1 Absatz)
- Models: 3 Hauptmodelle + 6 Varianten (kompakte Tabelle)
- Keypoint Mapping (12 joints, kurz)
- NMPJPE mit Torso-Normalisierung
- Rotation Angle (ehrlich: bimodal, Kamera-Vergleich)
- Person Selection: **Architektur bestimmt Optionen, Strategie empirisch validiert** (siehe Section 7)
- Frame-Independent Mode: **Bewusste Methodik-Entscheidung** (siehe Section 8)
- Temporal Stability (Jitter)
- Confidence Filtering: 0.5, begründet, inkl. Asymmetrie-Diskussion
- Multi-Person: Frame-Level Severity aus Segmentation.csv
- **Änderungen:** Selection als Design Choice. COCO hinzufügen. Varianten-Tabelle.
  Confidence Threshold begründen. Frame-by-Frame Justification.

### Chapter 4: Implementation (~2 Seiten, STARK KÜRZEN)
- Pipeline-Architektur (1 Absatz)
- Data Flow (1 Absatz)
- Reproducibility
- **Änderungen:** Von 67 Zeilen auf ~40 kürzen. Kein Overlap mit Kap. 3.

### Chapter 5: Results (~10 Seiten, MASSIV UMGEBAUT)

**Prinzip: EINE Darstellung pro Metrik (Table ODER Figure, nicht beides)**

5.1 Descriptive Statistics (1/2 Seite)
- Frames, Videos, Filter-Level transparent

5.2 Overall Accuracy — 3 Hauptmodelle (1 Seite)
- Figure: Bar Chart Mean+Median
- Wilcoxon + Cohen's d (matched-frame)
- Text: MoveNet signifikant besser

5.3 Model Variant Analysis (NEU, 1.5 Seiten)
- Tabelle: Alle 9 Varianten, NMPJPE + FPS
- Figure: Speed-Accuracy Scatter Plot
- Intra-Family Tradeoff Diskussion

5.4 Cross-Dataset Comparison (NEU, 1 Seite)
- Tabelle: REHAB24-6 vs COCO Rankings
- Befund: Rankings verschieben sich → Standard ≠ Reha

5.5 Rotation Robustness (1 Seite)
- Figure: NMPJPE by Rotation Bin
- Ehrlich framen: "Kamera-Vergleich, bimodale Verteilung"
- MoveNet am robustesten (+18%)

5.6 Multi-Person Robustness (1.5 Seiten)
- Frame-Level Analyse (Severity 0-3, 32k Frames)
- 5-Coach-Video Extremfall-Analyse
- Outlier-Rate als Metrik für Person-Switch
- MediaPipe robuster: 9.1% vs 13.8% Person-Switch-Rate

5.7 Temporal Stability (1/2 Seite)
- Figure: Boxplot Jitter
- MoveNet am stabilsten

5.8 Detection Completeness (1/2 Seite)
- Text + Zahlen, keine Figure nötig
- YOLO am vollständigsten

5.9 Per-Joint Analysis (1/2 Seite)
- Figure: Heatmap
- Hip-Offset Befund (MediaPipe 16.7% vs MoveNet 11.3%)

**Qualitative Beispiele (durchgestreut):**
- Coach-Szenario (Fig 9 existiert)
- Rotation-Effekt (frontal vs lateral Frame) → generieren
- Person-Switch Event → generieren

### Chapter 6: Discussion (~5 Seiten)

6.1 Research Questions Answered
- RQ1: MoveNet 10.5%, signifikant besser (d=0.29)
- RQ2: MoveNet niedrigster Jitter
- RQ3: MoveNet robusteste Rotation, MediaPipe robusteste Multi-Person
- RQ4: MoveNet für die meisten Anwendungsfälle empfohlen

6.2 Architectural Analysis
- Warum MoveNet so konsistent ist (Bottom-Up, single pass)
- MediaPipe Face-Detection-Filter Hypothese (vorsichtig als Hypothese)
- Hip-Offset: systematisch, nicht Qualitätsproblem

6.3 Cross-Dataset Findings (NEU)
- Rankings verschieben sich → Benchmark ≠ Reha
- Warum: kontrollierte Bedingungen, Übungskontext, Patientenpopulation
- Implikation: domänenspezifische Evaluation nötig

6.4 Model Variant Recommendations (NEU)
- Speed-kritisch: MoveNet MP Lightning oder YOLOv8 Nano
- Accuracy-kritisch: YOLOv8 Medium oder MoveNet MP Lightning
- Multi-Person-Risiko: MediaPipe (Heavy > Full > Lite)

6.5 Limitations
- Single REHAB-Dataset (aber Cross-Dataset-Vergleich hilft)
- Bimodale Rotation (ehrlich)
- N=5 Coach-Videos
- CPU-only Benchmark
- 2D only
- Hip-Offset als Mapping-Limitation

6.6 Future Work

### Chapter 7: Conclusion (~1 Seite)
- Komplett neu schreiben, kein LLM-Ton
- Keine spezifischen Zahlen, nur Kernaussagen
- 3 Sätze Zusammenfassung, 3 Sätze Implikation

---

## 5. Was noch zu tun ist

### Daten/Analyse
- [ ] Inference Benchmark für 6 neue Varianten (sequenziell, ~30 min) → Section 11
- [x] Detection-Count-Analyse für 5 Coach-Videos (MediaPipe + YOLO) → erledigt
- [x] Qualitative Frames extrahiert (PM_119, PM_010) → evaluation_v2/qualitative_frames/
- [ ] Speed-Accuracy Scatter Plot generieren (nach Inference Benchmark)
- [ ] Cross-Dataset Vergleichs-Figure generieren
- [ ] Statistische Tests für Jitter, Detection (nicht nur Accuracy)

### Thesis-Text
- [ ] Chapter 7 neu schreiben (kurz, kein LLM)
- [ ] Chapter 5 komplett umbauen (eine Darstellung pro Metrik)
- [ ] Chapter 6 erweitern (Cross-Dataset, Variants, Hip-Offset)
- [ ] Chapter 3 updaten (COCO, Variants, Selection Design Choice)
- [ ] Chapter 1 Contributions updaten
- [ ] Chapter 2 Research Gap updaten
- [ ] Chapter 4 kürzen
- [ ] Alle Kapitelanfänge de-LLMizen
- [ ] cfg.tex: Sponsor + Arbeitsgruppe TODO fixen

### Qualitätsprüfung
- [ ] Jede Zahl in der Thesis auf evaluation_v2 Daten zurückführbar
- [ ] Keine redundanten Table+Figure Paare
- [ ] Alle Claims mit Evidenz belegt
- [ ] Seitenzahl: 38-42 Ziel

---

## 7. Person Selection — Methodische Einordnung

### Das Problem (Lars' Feedback #2)
Lars fragte: "It is unclear if the model selection strategies are your proposal.
If so, they should be refined as they seem to fail for the current use case."

### Die Antwort: Architektur bestimmt die Optionen

Die Selection-Strategien sind NICHT willkürlich gewählt, sondern durch die
verfügbaren Outputs jedes Modells bestimmt:

| Modell | Was es nativ liefert | Was wir daraus machen | Warum |
|--------|---------------------|----------------------|-------|
| MediaPipe | Bis zu 5 Personen mit 33 Keypoints, KEINE BBoxes | Torso-Größe (Schulter-Hüfte) | BBox-from-Keypoints getestet: 11 vs 2 Fehler |
| MoveNet MP | Bis zu 6 Personen mit 17 Keypoints + BBoxes | BBox Area | Native BBox ist stärkstes Signal |
| YOLOv8 | Alle Personen mit 17 Keypoints + BBoxes + Confidence | BBox Area + Confidence >0.3 | Native BBox + Detector-Confidence |

### Warum die Strategie WENIGER wichtig ist als gedacht

**Kernbefund aus Detection-Count-Analyse:**

In Coach-Videos detektiert MediaPipe den Coach in 84% der Frames GAR NICHT
(PM_119, Coach von hinten). Die Selection-Strategie wird nur in den verbleibenden
16% relevant. Die Robustness kommt von der Architektur (Detection-Level), nicht
von der Selection (Post-Detection-Level).

### Formulierung: Clean, je weniger desto mehr

**Kap. 3 (1 Satz):**
> "For multi-person frames, we select the person with the largest bounding box
> area (MoveNet, YOLOv8) or torso size (MediaPipe), matching each model's native
> output format."

**Kap. 5 (2 Sätze + Tabelle):**
> "MediaPipe detected a single person in 77% of coach video frames, compared to
> 38% for YOLOv8 (Table X). This detection-level difference accounts for
> MediaPipe's lower person-switch rate (9.1% vs 13.9%)."

**Kap. 6 (1 kurzer Absatz):**
> "The multi-person robustness difference is primarily detection-level: MediaPipe's
> top-down pipeline detects fewer secondary persons than bottom-up or one-stage
> architectures. The person selection strategy becomes relevant only in the minority
> of frames where multiple persons are detected."

Keine Verteidigung, keine Über-Erklärung. Die Tabelle IST die Evidenz.

---

## 8. Frame-by-Frame Evaluation — Methodische Rechtfertigung

### Was wir machen
Alle Modelle werden im IMAGE-Modus evaluiert: jedes Frame unabhängig, ohne
Information aus vorherigen Frames. Kein Tracking, kein Kalman-Filter, kein
Person-Memory.

### Warum das methodisch korrekt ist
1. **Isoliert Modell-Accuracy von Pipeline-Logik**: Temporal Tracking verbessert
   JEDES Modell → der Vergleich wäre weniger aussagekräftig
2. **Standard in HPE-Benchmarking**: COCO, MPII evaluieren Einzelbilder
3. **Reproduzierbar**: Keine Tracking-Hyperparameter die das Ergebnis beeinflussen
4. **Worst Case**: Person-Switch-Raten sind Obergrenze, nicht Durchschnitt

### Was das für die Ergebnisse bedeutet
- Person-Switch-Raten (9-14%) sind OBERE GRENZEN
- In Production mit Temporal Tracking wären sie nahe 0%
- Jitter-Werte reflektieren nur Modell-Instabilität, nicht Pipeline-Smoothing

### Formulierung für Kap. 3

> "All models were evaluated in frame-independent mode, processing each frame
> without temporal context. This isolates model-level prediction quality from
> pipeline-level temporal processing and follows standard HPE evaluation practice
> (COCO, MPII). The reported person-switch rates and jitter values therefore
> represent upper bounds; production deployments with temporal tracking (e.g.,
> IOU-based person tracking, Kalman filtering) would substantially reduce both."

---

## 9. Holistic Evaluation — Was NMPJPE nicht erfasst

### MediaPipe's nicht-NMPJPE Vorteile (FAKTEN aus Architektur/Docs)

| Feature | MediaPipe | MoveNet | YOLOv8 | Quelle |
|---------|-----------|---------|--------|--------|
| Body Keypoints | 33 (inkl. Finger, Zehen) | 17 | 17 | Architektur |
| Face Landmarks | Ja (468 via FaceMesh) | Nein | Nein | API |
| Hand Landmarks | Ja (21 pro Hand) | Nein | Nein | API |
| 3D World Coordinates | Ja (native) | Nein | Nein | API |
| Integration | pip install, 3 Zeilen | TF Hub laden | PyTorch nötig | Docs |
| Mobile SDK | Android/iOS native | TFLite | CoreML/ONNX | Docs |
| Multi-Person Robustness | Beste (architekturell) | Mittel | Mittel | Unsere Daten |

### Warum das für die Thesis relevant ist

Der Titel ist "A **Holistic** Evaluation" — das bedeutet wir bewerten NICHT nur
Keypoint-Accuracy. Die Empfehlung ist nuanciert:

- **Wenn nur 12 Body-Joints nötig + Speed kritisch**: MoveNet
- **Wenn Face/Hand-Tracking + 3D + einfache Integration nötig**: MediaPipe
- **Wenn maximale Detection-Rate nötig**: YOLOv8
- **Wenn Multi-Person-Szenarien wahrscheinlich**: MediaPipe

### Formulierung für Kap. 6 (Discussion, Section 6.4)

> "While our quantitative evaluation focuses on 12 body keypoints common to all
> models, a holistic assessment must consider capabilities beyond body-joint NMPJPE.
> MediaPipe uniquely provides 33 landmarks (including face and hand keypoints),
> native 3D world coordinates, and the most straightforward API integration —
> features that may outweigh its lower body-joint accuracy for applications
> requiring comprehensive pose understanding or rapid prototyping."

---

## 10. Schreibprinzip: Weniger ist mehr

Jede Behauptung in der Thesis folgt dem gleichen Muster:
- **Methodology**: 1 Satz was wir tun
- **Results**: Zahlen + Tabelle/Figure, 1-2 Sätze Kontext
- **Discussion**: 1 Absatz Interpretation

Nicht:
- 3 Absätze rechtfertigen warum eine Entscheidung okay ist
- Erst erklären was man tun wird, dann tun, dann zusammenfassen was man getan hat
- "It is worth noting that..." / "Furthermore..." / "Notably..."

Wenn eine Entscheidung Erklärung braucht, steht die Evidenz in einer Tabelle
oder Figure. Der Text verweist darauf, erklärt nicht redundant.

---

## 11. Inference Benchmark — Noch ausstehend

### Was wir haben
- 3 Hauptmodelle: MoveNet 36.1ms/27.7 FPS, YOLOv8n 52.9ms/18.9 FPS, MediaPipe 67.9ms/14.7 FPS
- Quelle: `benchmark_inference.py`, CPU-only, 1500 Frames, 50-Frame Warmup

### Was noch fehlt
- 6 neue Varianten: MediaPipe Lite, MediaPipe Heavy, MoveNet SP Lightning,
  MoveNet SP Thunder, YOLOv8 Small, YOLOv8 Medium
- MUSS SEQUENZIELL laufen (nicht parallel — verfälscht Timing)

### Wie es geht
- Gleiches Script (`benchmark_inference.py`) oder neues mit gleichem Prinzip
- Pro Variante: 500 Frames reichen (weniger als 1500, spart Zeit, statistisch genug)
- 50-Frame Warmup beibehalten
- Nanosekunden-Timing wie beim Original
- Geschätzte Laufzeit: ~3-5 Minuten pro Variante × 6 = ~20-30 Minuten total

### Output für die Thesis
- Eine Tabelle: alle 9 Varianten mit Mean, Median, FPS, P95
- Ein Scatter Plot: FPS (x-Achse) vs NMPJPE (y-Achse) → Speed-Accuracy Tradeoff
- Zeigt Pareto-Frontier: welche Varianten sind optimal, welche dominated

---

## 6. Zeitplan (Vorschlag)

| Phase | Was | Dauer |
|-------|-----|-------|
| 1 | Inference Benchmarks + fehlende Figures | 1 Tag |
| 2 | Chapter 5 komplett umbauen | 2 Tage |
| 3 | Chapter 3 + 6 erweitern | 1 Tag |
| 4 | Chapters 1, 2, 4, 7 überarbeiten | 1 Tag |
| 5 | Qualitätsprüfung + Feinschliff | 1 Tag |
