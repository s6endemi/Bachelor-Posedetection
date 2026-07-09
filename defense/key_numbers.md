# Kernzahlen-Spickzettel — MUSS SITZEN

> Quelle: finale Arbeit, Kapitel 5 (`thesis/chapters/05_results.tex`).
> Merkstrategie: pro Dimension einen „Dreiklang" (MP / MoveNet / YOLO) memorieren.
> Reihenfolge im Kopf IMMER prüfen — sie wechselt pro Dimension (drei Profile!).

## Priorität 1 — die Story-Zahlen (ohne die geht nichts)

| Dimension | MediaPipe | MoveNet | YOLOv8 | Gewinner |
|---|---|---|---|---|
| **NMPJPE mean (valid)** | 12,53% | **10,52%** | 12,77% | MoveNet |
| **Displacement (Jitter-Proxy)** | 6,03% | **3,81%** | 5,22% | MoveNet |
| **Rotation frontal→lateral** | +27,5% | **+17,8%** | +31,7% | MoveNet |
| **Detection 12/12 komplett** | 55,1% | 38,2% | **79,1%** | YOLOv8 |
| **Coach-Outlier-Rate** | **9,1%** | 13,8% | 13,9% | MediaPipe |
| **FPS (CPU, Haupt-Konfig.)** | 14,7 | **27,7** | 18,9 | MoveNet |

**Der Drei-Profile-Satz:** MoveNet = Accuracy + Stabilität + Speed; YOLOv8 =
Completeness; MediaPipe = Coach-Robustheit. Kein universeller Sieger.

## Priorität 2 — Statistik & Aggregation

| Vergleich (Cluster-Ebene, n=10) | Δ mean | 95%-CI | p | p_Holm |
|---|---|---|---|---|
| MediaPipe − MoveNet | +1,01% | [0,62, 1,42] | 0,0020 | 0,0059 |
| MediaPipe − YOLOv8 | −0,49% | [−0,85, −0,12] | 0,0352 | 0,0352 |
| MoveNet − YOLOv8 | −1,50% | [−1,71, −1,27] | 0,0020 | 0,0059 |

- **p = 0,0020 = 2/1024** = Minimum → MoveNet gewinnt in ALLEN 10 Clustern (beide Male).
- Alle drei nach Holm signifikant. MoveNet #1 auf **allen 6 Übungen**.
- Frame-Kette: **367.200** (126 Seq. × 3 Modelle) → 363.074 (ohne Detection-Failures)
  → 354.879 (valid). Valid n pro Modell: 115.711 (YOLO) – 119.729 (MP).
- Frame-Gap MP−MoveNet 2,01pp ≠ Cluster-Differenz 1,01pp (Median-Aggregation dämpft
  MediaPipes schwere Fehler-Frames; Std 7,05 vs. 4,66).
- Viewpoint-Verteilung bimodal: ~40,8% frontal (0–20°), 17,5% Mitte, 41,6% lateral.

## Priorität 2 — Coach-Subset (5 Videos, 1.178 sampled frames, every 10th)

| | MediaPipe | MoveNet | YOLOv8 |
|---|---|---|---|
| 2+ Personen detektiert | 23% | 16% | **62%** |
| Coach-Outlier-Rate | **9,1%** | 13,8% | 13,9% |
| Coach mean NMPJPE (VOR Outlier-Cut!) | **45,4%** | 62,2% | 66,0% |

- **Der Kernbefund:** MoveNet exposed WENIGER Zweitpersonen als MediaPipe (16<23) und
  failt trotzdem öfter; MoveNet vs. YOLO: 4× Exposure-Unterschied, fast gleiche
  Outlier-Rate → **Detection count alone does not explain coach robustness.**
- Per Video: **PM_010-c17 bricht alle drei** (MP 81,9 / MoveNet 70,7 / YOLO 69,9);
  in den anderen 4 bleibt MediaPipe < 40% (38,0 / 17,2 / 24,1 / 30,9).
- Coach-Videos: PM_010, PM_011, PM_108, PM_119, PM_121 (alle c17).
- Coach-Figur: PM_119-c17 Frame 105 — MoveNet wählt dort den Therapeuten
  (largest-area), YOLO detektiert 3, wählt Patient; MediaPipe nur Patient.

## Priorität 2 — Failure-Mode-Signaturen

| Modell | Failures gesamt | Dominante Kategorie | Anteil |
|---|---|---|---|
| MediaPipe | 1.583 | Keypoint-Displacement | 69,6% |
| MoveNet | 1.875 | Confidence-Collapse | 51,4% |
| YOLOv8 | **5.610** | Missing-Detection | 63,8% |

- YOLO: 3.580 Frames ganz ohne Prediction ≈ **20× MediaPipe-Rate**.
- Reihenfolge der Taxonomie: Missing-Detection → Confidence-Collapse (<6/12 valide)
  → Multi-Person-Confusion (Outlier + severity≥2) → Keypoint-Displacement (Rest).
- Multi-Person-Confusion ≤ 11% überall. Robust für Cutoff 4–8.
- Merkhilfe: **top-down „commit-then-predict" → falsche confident Posen (KD);
  bottom-up per-Joint-Confidence → graceful degradation (CC); one-stage Kopplung →
  ganz oder gar nicht (MD).**

## Priorität 3 — Varianten & Cross-Dataset

**REHAB-Ranking (mean NMPJPE):** MoveNet_MultiPose 10,52 < YOLO_Medium 10,86 <
YOLO_Small 11,43 < MP_Heavy 11,51 < MP_Full 12,53 < YOLO_Nano 12,77 <
MoveNet_SP_Thunder 13,33 < MP_Lite 13,54 < MoveNet_SP_Lightning 14,97.

**COCO:** #1 YOLO_Medium 10,15; MoveNet_MultiPose nur #4 (12,44); MP_Full #8 (17,21);
MP stürzt generell ab, MoveNet-SP steigen (Thunder 7→3), YOLO stabil (Nano 6→6).
Single-Person-COCO ändert das Bild nicht. Hauptmodelle: MoveNet auf beiden #1,
MP/YOLO tauschen.

**Inference (CPU, ms → FPS):** SP_Lightning 9,9→**100,6** | MultiPose 36,1→27,7 |
SP_Thunder 36,2→27,6 | MP_Lite 45,7→21,9 | YOLO_n 52,9→18,9 | MP_Full 67,9→14,7 |
YOLO_s 97,1→10,3 | MP_Heavy 165,3→6,0 (P95: 304ms!) | YOLO_m 215,4→4,6.
Jitter-Top-3 = komplette MoveNet-Familie (3,31 / 3,61 / 3,81).

## Priorität 3 — Per-Joint / Hips

| Region | MP | MoveNet | YOLO |
|---|---|---|---|
| Schultern | **8,76** | 10,29 | 10,21 |
| Ellbogen | 11,24 | **10,00** | 13,08 |
| Handgelenke | 12,01 | **11,84** | 16,25 |
| **Hüften** | **16,72!** | **11,28** | 14,59 |
| Knie | 9,92 | **7,76** | 9,87 |
| Knöchel | 14,59 | **11,41** | 12,59 |

- MediaPipe gewinnt NUR Schultern. Hip-Ausschluss: MP 12,53→11,66 (MoveNet
  10,52→10,46) → **Gap 2,01pp → 1,21pp** = Hip-Anteil groß, aber nicht alles.

## Dataset-Basics (Blitz-Abruf)

10 Subjects (6m/4w, 25–50) · 6 Übungen · 65 → 63 Recordings (2 ohne beide Views+GT)
→ 126 Sequenzen · 1.072 Repetitionen · 30 fps, jeder 3. Frame = 10 Hz · GT: 41 Marker
→ 26 Joints → 2D-Projektion · 12 evaluierte Joints · c17 frontal / c18 lateral ·
COCO: 1.519 Bilder.
