# Chapter 4 — Implementation (Verteidigungsanalyse, kompakt)

> Quelle: `thesis/chapters/04_implementation.tex` (final, vollständig gelesen)
> Geringe Angriffsfläche — aber zwei starke Q&A-Assets (Reproduzierbarkeit, Erweiterbarkeit).

## 1. Kernaussagen

1. **Zwei-Phasen-Architektur:** Phase 1 = teure Inferenz, einmal pro Modell auf allen
   Recordings + COCO-Subset → speichert per-Frame-Predictions + Confidences als
   komprimierte **.npz**-Arrays. Phase 2 = lädt Predictions + GT, wendet alle
   Filter-/Aggregations-/Statistikregeln an, erzeugt alle Metriken.
   **Warum:** Filter-, Aggregations- oder Statistik-Entscheidungen können revidiert
   werden, ohne die Inferenz zu wiederholen. (Genau das ist passiert: Die
   Cluster-Statistik kam als Revision — ohne Re-Inferenz.)
2. **Ein Interface für alle 9 Konfigurationen:** Abstrakte `PoseEstimator`-Basisklasse
   wrappt die drei fundamental verschiedenen APIs (MediaPipe Tasks API / MoveNet via
   TF Hub mit manuellem Padding/Unpadding / YOLOv8 Ultralytics). Downstream-Code
   **brancht nie auf Modelltyp**.
3. **Bewusst dünne Abstraktion:** Estimatoren normalisieren nur Formate (33→17 etc.);
   die Evaluationslogik **inkl. Person-Selection bleibt zentral in Phase 2** → keine
   in Wrappern versteckten Entscheidungen, garantierte Konsistenz über Modelle.
4. **Datenformat:** Array-Shape **(N, 17, 3)** im COCO-Index-Raum — N gesampelte
   Frames, 17 Keypoints, 3 = (x, y, confidence) — plus Rotationswinkel-Array pro Frame.
   Gespeichert werden 17 (gemeinsamer Index-Raum), evaluiert 12 (Auswahl in Phase 2).
5. **Reproduzierbarkeit:** Jede Headline-Zahl traceable zu einem Phase-2-Artefakt;
   separates **Provenance-File** mappt jeden berichteten Wert auf seine Quelldatei.
   Phase 2 ist **deterministisch** auf den gespeicherten .npz. Ehrliche Einschränkung
   im Text: Determinismus über verschiedene Phase-1-Runtimes ist nicht separat
   garantiert.
6. Inference-Benchmark läuft isoliert, sequenziell, mit high-resolution Timern.

## 2. Q&A-Assets

**"How do you ensure your numbers are reproducible?"**
> "Two-phase design: inference runs once and stores per-frame predictions as .npz;
> everything downstream — filtering, aggregation, statistics — is deterministic on
> those artifacts. Every headline number maps to a source file via a provenance file.
> Re-running phase 2 reproduces the reported metrics exactly; what I don't separately
> guarantee is bit-level determinism across different inference runtimes, and the
> thesis says so."

**"What would it take to add another model, say RTMPose?"** (Anschluss an Modellwahl-Frage!)
> "One new estimator class implementing the same interface — load weights, return
> native detections, map to the shared COCO index space. Phase 2, including selection,
> filtering and statistics, runs unchanged. The pipeline was explicitly built for that."

**"Why store 17 keypoints if you evaluate 12?"**
> "The COCO index space is the natural shared format of two of the three models;
> storing it keeps phase 1 model-faithful, and the 12-joint evaluation subset is a
> phase-2 decision that can be revised without re-running inference."

**Detail-Wissen (falls gefragt):** MoveNet braucht manuelles Padding/Unpadding, weil
das TF-Hub-Modell feste Input-Constraints hat (Aspect-Ratio/Größe) — die Rohframes
werden gepaddet und die Koordinaten zurückgerechnet.

## 3. Folien-Kandidat

- Halbe Folie: Pipeline-Diagramm *Videos → Phase 1 (9 Modelle, ein Interface) → .npz →
  Phase 2 (Filter → Metriken → Statistik) → Artefakte/Provenance*. Kombinierbar mit
  der Protokoll-Folie aus Kapitel 3.
