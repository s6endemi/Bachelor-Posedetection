# Bachelor Thesis — Aktueller Stand

> Beschreibender Statusbericht. Enthält was existiert, nicht was als nächstes zu tun ist.
> Aktueller Stand nach Session 3.
> Für Session-3-Zusammenfassung siehe `HANDOFF_REVIEWER_V3.md` im Projekt-Root.
> Für historischen Kontext (geschlossene Grundentscheidungen) siehe `HANDOFF_REVIEWER_V2.md`.

---

## Einstiegsreihenfolge

Wenn du neu in die Arbeit einsteigst, empfiehlt sich diese Lesereihenfolge:

1. `HANDOFF_REVIEWER_V3.md` (Projekt-Root) — Was Session 3 gemacht hat, aktuelle Kern-Entscheidungen
2. Dieses Dokument (`evaluation_v2/STATUS.md`) — objektiver Stand, Artefakte, Dateipfade
3. `HANDOFF_REVIEWER_V2.md` (Projekt-Root) — historischer Kontext, geschlossene Grundentscheidungen aus Session 2
4. `thesis/chapters/*.tex` und `thesis/abstract.tex` — aktuelle Kapiteltexte
5. `HANDOFF_PROMPT.md` — Originales Betreuer-Feedback von Lars Doorenbos
6. `evaluation_v2/THESIS_REWRITE_PLAN.md` — historische Planung, Claims-Framework (als Referenz)
7. `WRITING_GUIDE.md` — Schreibrichtlinien

---

## Stand in drei Sätzen

1. Alle sieben Kapitel, das Abstract und Appendix A sind geschrieben und stilistisch durchpoliert. In Session 3 wurde zusätzlich eine Failure-Mode-Taxonomie als neue analytische Erweiterung integriert (Sec 5.11 + Discussion-Paragraph in 6.2).
2. Die Auswertungsartefakte sind eingefroren und liegen vollständig in `evaluation_v2/results/`, inkl. Failure-Mode-Klassifikation (`failure_mode_*`), Per-Exercise-Breakdown, cluster-robuste Statistik, normalisierte Frame-to-frame-Prediction-Displacement-Metrik, Hip-Sensitivity, Coach-Detection-Counts, COCO-Ergebnisse, Speed-Benchmark.
3. Der Build ist sauber (61 Seiten, 44 Textteil, 0 Errors, 0 undefined references); `cfg.tex` ist finalisiert. Lars hat grünes Licht zur Abgabe gegeben, Eren bereitet Abgabe-Mail vor.

---

## Kapitel-Status

| Kapitel | Datei | Letzte Änderung (Session 3) |
|---------|-------|------------------|
| Abstract | `thesis/abstract.tex` | komplett neu strukturiert, 16 Mikro-Änderungen, Failure-Mode-Satz ergänzt |
| 1 Introduction | `thesis/chapters/01_introduction.tex` | Doppelsatz aufgelöst, LLM-Füller raus, neue 6. Contribution „Failure-mode taxonomy" |
| 2 Background | `thesis/chapters/02_background.tex` | 6 Stil-Fixes (still-thin, broader landscape, meta-narration) |
| 3 Methodology | `thesis/chapters/03_methodology.tex` | atan2-Notation, Coach-Subset-IDs explizit, MC-Permutation-Kontext, Meta-Opener gestrichen |
| 4 Implementation | `thesis/chapters/04_implementation.tex` | Meta-Opener raus, estimator-interface-Satz aktiviert |
| 5 Results | `thesis/chapters/05_results.tex` | Per-Exercise-Satz, n-Disclaimer, „primary inferential unit" vereinfacht, **neue Sec 5.11 Failure-Mode Decomposition** |
| 6 Discussion | `thesis/chapters/06_discussion.tex` | „most interesting"→„central", „one of cleanest"→„a clean", Bridge-Sätze 6.6/6.9 raus, **neuer \paragraph „Failure-mode signature" in Sec 6.2** |
| 7 Conclusion | `thesis/chapters/07_conclusion.tex` | „Two findings dominate"→„stand out", „bounded in important ways"→„Three limitations matter", Deployment-Paragraph restrukturiert, Failure-Mode-Satz ergänzt |
| Appendix A | `thesis/chapters/appendix.tex` | **neu erstellt** — A.1 Per-Exercise Accuracy, A.2 Failure-Mode Counts |

---

## Figuren

**`thesis/fig/coach_detection_comparison.png` / `.jpg`** — vierteilige Coach-Figur auf PM_119-c17 Frame 105.
- Farbkonvention: MediaPipe grün, MoveNet rot, YOLOv8 blau; eine Farbe pro Modell über alle Panels.
- Native Bounding Boxes nur auf vom Modell selektierter Person. MediaPipe hat keine (bottom-up/top-down-spezifisch).
- Keypoints für alle detektierten Personen.
- In Kap. 5 Sec 5.6 referenziert, in Kap. 6 Sec 6.3 in der MediaPipe--MoveNet-Diskussion erneut aufgegriffen.

**`thesis/fig/failure_mode_taxonomy.pdf`** — Stacked-Bar-Plot der vier Failure-Kategorien pro Modell.
- Farben: Missing-Detection grau, Confidence-Collapse blau, Multi-Person-Confusion orange, Keypoint-Displacement rot.
- In Kap. 5 Sec 5.11 als `fig:failure-modes` referenziert.

---

## Evaluations-Pipeline

`evaluation_v2/` enthält den aktuellen Pipeline-Stand:

```
evaluation_v2/
├── config.py                     Zentrale Konfiguration (Thresholds, Pfade)
├── evaluate.py                   REHAB24-6 Evaluation (Hauptmodelle)
├── evaluate_all.py               Alle 9 Varianten auf REHAB24-6
├── analyze.py                    Statistiken und Figuren
├── coco_evaluate_parallel.py     COCO val2017 parallel für alle Varianten
├── rehab_variant_inference.py    Variant-Inferenz auf REHAB24-6
├── hip_sensitivity.py            Hip-Offset-Analyse
├── patient_level_stats.py        Cluster-robuste Statistik (RNG_SEED = 20260412)
├── failure_mode_analysis.py      NEU (Session 3): Taxonomie für 3 Hauptmodelle
├── failure_mode_variants.py      NEU (Session 3): 9-Varianten-Consistency-Check (Defense-Pocket)
├── per_exercise_breakdown.py     NEU (Session 3): Per-Übung-Aggregation
├── STATUS.md                     Dieses Dokument
├── THESIS_REWRITE_PLAN.md        Historische Planung, Claims-Framework
└── results/                      Canonical artifacts
    ├── canonical_provenance.md   Master-Verweise für alle Zahlen
    ├── rehab24/                  Hauptmodelle, Frame-Level + Cluster-Stats + Jitter
    ├── rehab24_all/              Alle 9 Varianten auf REHAB24-6
    ├── coco/                     COCO val2017 Ergebnisse
    ├── coach_detection_counts_summary.csv / .json
    ├── hip_sensitivity_summary.csv / .json
    ├── dataset_structure_audit.md
    ├── inference_benchmark_all.csv        Speed-Benchmark aller 9 Varianten
    ├── failure_mode_summary.csv / .json   NEU (Session 3): Hauptergebnis Taxonomie
    ├── failure_mode_per_video.csv         NEU (Session 3): Breakdown pro Video
    ├── failure_mode_sensitivity.csv       NEU (Session 3): Threshold-Sensitivity
    ├── failure_mode_variants.csv          NEU (Session 3): 9-Varianten-Consistency
    ├── failure_mode_taxonomy.png / .pdf   NEU (Session 3): Stacked-Bar-Plot
    ├── per_exercise_nmpjpe.csv            NEU (Session 3): NMPJPE pro Übung × Modell
    ├── per_exercise_jitter.csv            NEU (Session 3): Prediction-Displacement pro Übung × Modell
    └── per_exercise_*_heatmap.png         NEU (Session 3): Heatmap-Plots (nicht in Thesis)
```

Alle Zahlen, die in der Thesis vorkommen, sind laut `canonical_provenance.md`
auf eine dieser Dateien zurückführbar. Ältere Ergebnisse in `analysis/` und
`docs/04_RESULTS.md` sind überholt und sollten nicht mehr als Quelle verwendet
werden.

---

## Dataset-Fakten (häufig übersehen)

REHAB24-6 (Čerňek et al., 2025):

- 10 healthy adult volunteers (6 männlich, 4 weiblich, Alter 25--50).
- Rekrutierung als *early adopters of technology*, keine klinischen
  Einschlusskriterien.
- Physiotherapeut-geführte Aufnahmen mit korrekten und absichtlich fehlerhaften
  Ausführungen.
- 65 Recordings, 1072 annotierte Repetitionen, 2 RGB-Kameras bei 30 FPS,
  41-Marker OptiTrack-Motion-Capture.
- Die Dataset-Autoren selbst: *the inclusion of real patient data would provide
  substantially greater clinical relevance.*

Konsequenz: die Thesis verwendet durchgehend *rehabilitation-oriented proxy
benchmark* und *10 healthy adult volunteers*, keine Formulierungen wie
*21 patients* oder *clinical cohort*.

---

## Statistische Methodik

- Primäre Inferenzebene: Subject-Cluster (n=10) aus dem `person_id` der
  Segmentation-Metadaten.
- Paired sign-flip permutation test auf cluster-wise Differenzen, primär exakt
  über alle 1024 Sign-Assignments enumeriert, mit Holm-Korrektur und
  Bootstrap-Konfidenzintervallen.
- Video-Level und Frame-Level dienen nur als Sensitivity-Analysen.
- COCO-Ergebnisse werden als deskriptive Transferability-Analyse geführt, nicht
  als inferentielle Aussage auf REHAB-Ebene.

Ergebnisse auf Cluster-Ebene (aus `patient_level_stats.json`):

- MoveNet vs MediaPipe: p = 0.0020, Holm p = 0.0059
- MoveNet vs YOLOv8: p = 0.0020, Holm p = 0.0059
- MediaPipe vs YOLOv8: p = 0.0352

RNG-Seed für Bootstrap und die Video-Level-MC-Sensitivity: `RNG_SEED = 20260412` (`patient_level_stats.py:32`).

---

## Failure-Mode-Taxonomie (Session 3)

Mutually-exclusive Vier-Kategorien-Klassifikation aller Failure-Frames:

| Modell | Gesamt-Failures | Dominante Mode | Anteil |
|---|---|---|---|
| MediaPipe | 1 583 (1,29%) | Keypoint-Displacement | 69,6% |
| MoveNet | 1 875 (1,53%) | Confidence-Collapse | 51,4% |
| YOLOv8 | 5 610 (4,58%) | Missing-Detection | 63,8% |

Dominanz ist robust über Collapse-Threshold 4/6/8. Quelle: `failure_mode_summary.csv`.

In Thesis verwendet: Sec 5.11 (Observation), Sec 6.2 „Failure-mode signature"-Paragraph (Interpretation mit n=1-Limitation).

Varianten-Consistency (`failure_mode_variants.csv`): MediaPipe und YOLOv8 familienweit konsistent; MoveNet SinglePose shiftet zu Missing-Detection wegen Output-Contract. **Dieses Ergebnis bewusst nicht im Text.** Defense-Pocket.

---

## Nicht mehr verwendete Dateien

- `analysis/` — frühere Evaluation-Scripts, ersetzt durch `evaluation_v2/`.
- `docs/04_RESULTS.md`, `docs/03_EXPERIMENTS.md` — Zwischenstände, nicht mehr
  kanonisch.
- `docs/REFERENCE_ANALYSIS.md` — historische Problem- und Diskrepanzanalyse,
  bleibt als Referenz erhalten, ist aber kein Quellenbeleg mehr für aktuelle
  Zahlen.
