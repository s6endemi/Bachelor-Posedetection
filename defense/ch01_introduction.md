# Chapter 1 — Introduction (Verteidigungsanalyse)

> Quelle: `thesis/chapters/01_introduction.tex` (final, vollständig gelesen)

## 1. Kernaussagen (Rückgrat)

1. **Motivationskette:** Adhärenz-Problem in Heim-Physiotherapie (Jack et al. 2010) →
   Telerehabilitation braucht Remote-Monitoring/Feedback (Simmich et al. 2024) → HPE aus
   RGB ermöglicht das ohne Marker-Systeme/Spezialhardware (Hellsten et al. 2021) →
   Modellwahl ist die offene Frage.
2. **Vier Deployment-Constraints**, die Standard-Benchmarks (COCO, MPII) nicht abdecken:
   - **Camera positioning** — schräge/laterale Orientierung, Selbst-Okklusion
   - **Multi-person interference** — Angehörige/Therapeut im Bild, Target-Retention
   - **Temporal stability** — Frame-zu-Frame-Stabilität für Echtzeit-Feedback
   - **Resource constraints** — CPU-only, keine GPU zu Hause
   → Diese vier mappen 1:1 auf die Evaluationsdimensionen (Story-Gerüst für Folien!).
3. **Gap (zweiteilig, gescoped):** (a) MoveNet nie auf Reha-Benchmark mit MoCap-GT
   verglichen; (b) keine Arbeit kombiniert alle sechs Dimensionen in einer
   Mobile-Model-Evaluation. Absicherung: *"In the literature reviewed in Chapter 2"*.
4. **5 Research Questions** — Merksatz: *how good (RQ1: accuracy+completeness), how
   stable (RQ2), how robust (RQ3: viewpoint+multi-person), does it transfer (RQ4:
   COCO), what to deploy (RQ5)*.
5. **7 Contributions:** (1) Reha-Evaluation 3 Familien + 6 Varianten, (2)
   Multi-Metrik-Benchmark, (3) Subject-Cluster-Inferenz (Permutation + Holm +
   Bootstrap), (4) Cross-Dataset-Transferanalyse (9 Varianten auf COCO val2017), (5)
   Multi-Person-Dekomposition (segmentation-defined Subset + coach extreme-case Subset),
   (6) Failure-Mode-Taxonomie (Missing-Detection, Confidence-Collapse,
   Multi-Person-Confusion, Keypoint-Displacement — jedes Modell von einer Kategorie
   dominiert), (7) Deployment-Guidance.

## 2. Memorier-Fakten

- Referenzen: **Jack et al. 2010** (Adhärenz-Barrieren), **Simmich et al. 2024**
  (Telerehab), **Hellsten et al. 2021** (HPE-Potenzial für Physio), **Lin et al. 2014**
  (COCO), **Andriluka et al. 2014** (MPII).
- Formulierung "normalized frame-to-frame prediction displacement" = die Jitter-Metrik
  (bewusst neutral benannt, kein "jitter" als Begriff im Claim).
- 12-joint shared skeleton als gemeinsamer Nenner.

## 3. Design-Entscheidungen + Verteidigung

| Entscheidung | Ehrlich | Strategisch |
|---|---|---|
| 3 Modelle (MediaPipe, MoveNet, YOLOv8-Pose) | mobile-taugliche, produktionsreife Ökosysteme, CPU-Echtzeit | decken die **drei Architektur-Paradigmen** ab (top-down / bottom-up / one-stage) → Unterschiede architektonisch attribuierbar |
| "Holistic" im Titel | klingt groß | im Abstract **operational definiert** als konkrete Qualitätsdimensionen |
| Gap gescoped ("in the literature reviewed") | man kennt nie alle Literatur | falsifizierbar nur innerhalb des Reviews — sauber |
| 2D statt 3D | Modelle sind primär 2D | bewusste Scope-Entscheidung: 2D = gemeinsamer Nenner; 3D-Lifting wäre eigene Arbeit; Limitation in Kap. 6 |
| 5 RQs statt Hypothese | Evaluationsarbeit | RQs mappen 1:1 auf Sektionen und werden in 6.1 explizit beantwortet — geschlossener Kreis |

## 4. Gefährliche Fragen + Musterantworten (EN)

**Q1: "Why these three models? Why not RTMPose, ViTPose, or OpenPose?"**
> "I selected models by three criteria: real-time capability on consumer CPUs, mature
> mobile deployment ecosystems, and joint coverage of the three main architectural
> paradigms — MediaPipe as a top-down detector-tracker pipeline, MoveNet as a bottom-up
> center-based model, YOLOv8-Pose as a one-stage detector. ViTPose targets server-scale
> deployment; OpenPose is unmaintained and slow on CPU. Paradigm coverage is what lets
> Chapter 6 attribute behavioral differences to architectural choices."

⚠️ Follow-up-Gefahr: **RTMPose ist CPU-tauglich.** Zweite Verteidigungslinie:
Auswahlkriterium war Paradigmen-Abdeckung + Deployment-Reife zum Projektstart, nicht
Marktvollständigkeit; Pipeline ist via Estimator-Interface erweiterbar (Kap. 4).

**Q2: "You motivate smartphones but benchmark on a desktop CPU — a mismatch?"**
> "A deliberate proxy choice, named as a limitation. Desktop-CPU inference is a
> reproducible, hardware-independent bound for the deployment class 'no dedicated
> accelerator'. Absolute latencies differ on phones, but the thesis claims relative
> ordering, driven by model complexity. On-device benchmarking is the first item of
> future work."

**Q3: "How confident are you in the gap claim (MoveNet never evaluated on rehab + MoCap)?"**
> "The claim is scoped to the literature reviewed in Chapter 2 — HPE-in-rehabilitation
> surveys plus the REHAB24-6 benchmark papers. Within that scope, MoveNet does not
> appear in any MoCap-validated rehabilitation evaluation. I deliberately phrased it as
> a scoped statement, not a universal one."

**Q4: "Why is temporal stability a *model* property if real systems smooth anyway?"**
> "Because I evaluate models, not pipelines. Temporal post-processing can be applied
> uniformly on top of any model; what differs between models is raw per-frame stability,
> which my displacement metric isolates. An inherently stable model needs less smoothing
> and adds less feedback latency — which matters for real-time exercise feedback."

**Q5: "What is actually novel here?"**
> "Three things: six quality dimensions under one protocol on rehabilitation-style data
> with MoCap ground truth; a subject-cluster statistical treatment with paired
> permutation tests, which most comparisons skip; and a failure-mode taxonomy showing
> each architecture fails in a characteristically different way. The finding that the
> models form three profiles rather than one ranking is itself deployment-relevant."

## 4b. Verständnis-Klärungen (Session 6.7.2026)

- **Fine-Tuning / Modellwahl / 2D-vs-3D:** ausführlich geklärt → `qa_catalog.md` A1–A3
  (Fundament + EN-Antworten). Kern: off-the-shelf = Deployment-Frage + TFLite-Artefakte
  nicht trainierbar; Modellwahl kriteriengeleitet (CPU-Echtzeit, Ökosystem-Reife,
  Paradigmen-Abdeckung) + Ablehnungsgründe je Alternative; 2D = gemeinsamer Nenner,
  nur MediaPipe hat natives (hüft-zentriertes) 3D.
- **Statistik-Trio (paired permutation, Holm, Bootstrap):** Fundament Schritt für
  Schritt → `ch03_methodology.md` §7; EN-Antworten → `qa_catalog.md` B1–B5.

## 5. Folien-Kandidaten aus Kapitel 1

- **Folie "Motivation":** Adhärenz-Problem → Telerehab → HPE als Sensor (1 Bild, 3 Sätze).
- **Folie "Why standard benchmarks are not enough":** die 4 Deployment-Constraints als
  Quadrant/Icons — direkt danach: "each constraint becomes an evaluation dimension".
- **Folie "Research Questions":** max. 5 Zeilen, RQ1–RQ5 als Kurzformen.
- **Folie "Contributions"** optional — bei 30 Min besser in die Ergebnis-Story integrieren
  statt als Aufzählung.
