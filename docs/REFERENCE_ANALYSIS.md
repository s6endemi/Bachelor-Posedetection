# Strukturierte Referenz-Analyse der Bachelorarbeit

> **Zweck:** Dieses Dokument dient als Standalone-Referenz während der Überarbeitung.
> Es enthält exakte Zahlen, Code-Referenzen und alle identifizierten Diskrepanzen.
>
> **Erstellt:** 2026-04-03 durch systematische Analyse aller Quellen.

---

## A. Problem-Lösung-Referenz

### Problem 1: MediaPipe Detection Failures (29%)

- **Problem:** MediaPipe lieferte bei 29% aller Frames keine Keypoints (leere `(0,0)` Rückgabe).
- **Ursache:** Default `min_detection_confidence=0.5` zu strikt — gute Detections mit Confidence 0.45 wurden verworfen.
- **Lösung:** `min_detection_confidence=0.1` gesetzt.
- **Impact:** Detection Failures: 29% → 0.4%; NMPJPE: 128% → ~10%.
- **Code:** `src/pose_evaluation/estimators/mediapipe_estimator.py`, Zeile 42: `min_detection_confidence: float = 0.1` (Konstruktor-Default). Zeile 86: `min_pose_detection_confidence=self.min_detection_confidence` (Option an PoseLandmarker).
- **Thesis-Kapitel:** Kap. 3 (Section 3.2 Models Under Evaluation, MediaPipe-Paragraph); Kap. 6 implizit.

---

### Problem 2: YOLO wählt falsche Person (37% NMPJPE bei 50-60°)

- **Problem:** YOLO hatte 37% NMPJPE bei 50-60° Rotation — eine Hintergrund-Person wurde selektiert.
- **Ursache:** Code nutzte `keypoints.xy[0]` (erste Person) statt die größte Person.
- **Lösung:** BBox Area Selection — größte Bounding Box wird gewählt.
- **Impact:** NMPJPE bei 50-60°: 37% → ~14%; Wrong Person: häufig → 0%.
- **Code:** `src/pose_evaluation/estimators/yolo_estimator.py`, Zeilen 76-90: Multi-Person-Logik mit `boxes.xyxy[i]` und Area-Berechnung.
- **Thesis-Kapitel:** Kap. 3 (Section 3.6 Person Selection Strategies, BBox-Paragraph); Kap. 5 (Section 5.4 Multi-Person); Kap. 6 (Section 6.3 Person Selection).

---

### Problem 3: MoveNet SinglePose kann keine Person wählen

- **Problem:** MoveNet SinglePose liefert nur 1 Person — bei Hintergrund-Person wird manchmal die falsche getrackt (32% NMPJPE bei 40-50°).
- **Ursache:** Architekturelle Limitation: SinglePose hat keine Multi-Person-Fähigkeit, daher keine nachträgliche Selection möglich.
- **Lösung:** Wechsel zu MoveNet MultiPose Lightning (bis zu 6 Personen, mit BBox und Score).
- **Impact:** Wrong Person: 2.8% → 0%.
- **Code:** `src/pose_evaluation/estimators/movenet_multipose_estimator.py` (gesamte Datei ist die Lösung). Zeilen 80-93: BBox Area Selection mit `score > 0.1` Filter.
- **Thesis-Kapitel:** Kap. 3 (Table 3.2 Model Configs — MultiPose Lightning); Kap. 6 (Architektur-Vergleich).

---

### Problem 4: BBox funktioniert nicht für MediaPipe

- **Problem:** Versuch, MediaPipe auch auf BBox-Selection umzustellen, ergab MEHR Fehler.
- **Ursache:** MediaPipe liefert keine echten Bounding Boxes. Pseudo-BBox aus Keypoint-Extremen misst "Spread" (Armposition) statt "Size" (Kamera-Distanz).
- **Lösung:** Torso-Größe (Schulter-Hüfte-Distanz) als Selection-Metrik für MediaPipe.
- **Impact:** Multi-Person Selection Errors: 11 (BBox) → 2 (Torso).
- **Code:** `src/pose_evaluation/estimators/mediapipe_estimator.py`, Zeilen 127-138: `if len(results.pose_landmarks) > 1:` → Torso-Size-Vergleich über `shoulder_y` und `hip_y`.
- **Thesis-Kapitel:** Kap. 3 (Section 3.6 Person Selection, Torso-Paragraph); Kap. 5 (Table 5.5 Torso Size Analysis); Kap. 6 (Section 6.3).

---

### Problem 5: Torso-Selection für YOLO (falscher Ansatz)

- **Problem:** Torso-Selection wurde zunächst auch für YOLO versucht, funktionierte aber nicht zuverlässig.
- **Ursache:** Bei YOLO sind Torso-Größen oft ähnlich (20% Differenz), aber BBox-Flächen deutlich verschieden (230% Differenz). BBox ist das stärkere Signal.
- **Lösung:** Echte BBox für YOLO/MoveNet, Torso nur für MediaPipe.
- **Impact:** Keine direkten Zahlen — bestätigt dass modellspezifische Strategien nötig sind.
- **Code:** Keine separate Code-Stelle; manifestiert sich darin, dass `yolo_estimator.py` BBox und `mediapipe_estimator.py` Torso nutzt.
- **Thesis-Kapitel:** Kap. 3 (Section 3.6, einleitender Absatz über modellspezifische Strategien).

---

### Problem 6 (Erkenntnis): MoCap Rotationswinkel-Variation

- **Problem/Beobachtung:** Rotationswinkel variiert ±2.5° selbst bei "statischer" Pose.
- **Ursache:** Armbewegungen verschieben Schultermarker (~10cm in z-Richtung); MoCap-Systemrauschen ±1-2cm; natürliche Körperschwankungen.
- **Lösung:** 10°-Bins absorbieren die Variation. Kein Bug, physikalische Realität.
- **Impact:** Kein numerischer Impact auf Ergebnisse — bestätigt Methodenwahl.
- **Code:** Nicht direkt im Code; relevant für `src/pose_evaluation/utils/rotation.py` (arctan2-Formel) und `src/pose_evaluation/inference/pipeline.py`, Zeile 86: `np.degrees(np.arctan2(abs(dz), abs(dx)))`.
- **Thesis-Kapitel:** Kap. 3 (Section 3.5 Rotation Angle Computation, letzter Absatz über Bin-Width-Wahl).

---

### Problem 7 (Kamera-Koordinatensystem): KRITISCH

- **Problem:** Berechnete Rotationswinkel ergaben keinen Sinn — frontal aussehende Personen hatten falsche Winkel.
- **Ursache:** Kameras sind nicht entlang der MoCap-Achsen ausgerichtet. Camera 17 schaut aus ~65° zum MoCap-System; Camera 18 ist 90° zu Camera 17 gedreht.
- **Lösung:** Empirische Offset-Transformation: `θ_c17 = |θ_raw - 65°|`, `θ_c18 = 90° - θ_c17`.
- **Impact:** Vorher: Winkel-Fehler-Korrelation sinnlos. Nachher: 0-20° frontal → 8-11% NMPJPE; 70-90° seitlich → 15-18% NMPJPE.
- **Code:** `src/pose_evaluation/inference/pipeline.py`, Zeilen 47-48: `C17_FRONTAL_OFFSET = 65.0`. Zeilen 67-97: `calculate_rotation_angle()` Methode.
- **Thesis-Kapitel:** Kap. 3 (Section 3.5 Rotation Angle Computation, Equations 3.3 und 3.4).

---

### Problem 8 (Confidence-Filter): MediaPipe Unterkörper-Ausreißer

- **Problem:** ~22% der MediaPipe-Frames hatten extrem hohe Fehler (>100px) am Unterkörper.
- **Ursache:** MediaPipe lieferte niedrige Joint-Confidence (0.003-0.07) bei schwierigen Posen — diese Keypoints waren unbrauchbar, wurden aber in die Evaluation einbezogen.
- **Lösung:** Per-Joint Confidence-Filter: Joints mit Confidence < 0.5 werden von NMPJPE-Berechnung ausgeschlossen.
- **Impact:** MediaPipe NMPJPE: 18.3% → 12.9% (ohne Filter → mit Filter); 6.8% der MediaPipe-Joints gefiltert, 2.2% MoveNet, 0% YOLO.
- **Code:** `src/pose_evaluation/evaluation/evaluator.py`, Zeile 58: `MIN_JOINT_CONFIDENCE = 0.5`. Zeilen 183-185: `valid_mask = pred_confidences >= self.MIN_JOINT_CONFIDENCE; filtered_errors = np.where(valid_mask, normalized_errors, np.nan)`.
- **Thesis-Kapitel:** Kap. 3 (Section 3.8 Confidence Filtering and Outlier Handling); Kap. 5 (Table 5.1 Detection Summary).

---

### Problem 9: Frame-Step-Bug in alter Evaluation

- **Problem:** Alte Evaluationszahlen waren falsch (z.B. YOLO c17: 54.9% statt tatsächlich 24.6%).
- **Ursache:** Alter Code berücksichtigte `frame_step=3` nicht korrekt — GT-Frame-Index wurde nicht mit Step multipliziert.
- **Lösung:** Neues Evaluationsscript `run_evaluation.py` (jetzt in `archive/old_scripts/`) mit korrektem Frame-Alignment: `gt_idx = i * frame_step`.
- **Impact:** Alle Zahlen vor dem 09.01.2026-Fix sind ungültig. Aktuelle kanonische Zahlen stammen aus der Neu-Evaluation.
- **Code:** `archive/old_scripts/run_evaluation.py` (fixer Evaluator). Aktuell: `src/pose_evaluation/evaluation/evaluator.py` Zeile 234: `for frame_idx in range(min(num_frames, len(gt_2d), len(rotation_angles)))` — hier wird die Prediction direkt mit dem gleichindizierten GT-Frame verglichen, da Predictions bereits gestepped gespeichert sind.
- **Thesis-Kapitel:** Nicht explizit erwähnt (internes Bugfix).

---

## B. Zahlen-Referenz mit Filterungs-Level

### B.1 Die drei Filterungs-Level

| Level | Beschreibung | Videos | Frames pro Modell | Anwendung |
|-------|-------------|--------|-------------------|-----------|
| **All Data** | Alle 126 Videos, alle Frames | 126 | ~118K | VERALTET — nicht verwenden |
| **Clean 121** | 121 Videos ohne Coach, MIT Ausreißern | 121 | ~115K | Multi-Person-Analyse (Baseline) |
| **Clean + Outlier Removed** | 121 Videos, Frames >100% NMPJPE entfernt | 121 | ~114K | **PRIMÄRE Accuracy-Zahlen** |

### B.2 Primäre Accuracy (Clean + Outlier Removed)

**Quelle:** Thesis Table 5.1 (`thesis/chapters/05_results.tex`, Zeile 50-54). HANDOFF_PROMPT Section 6.

| Modell | Mean NMPJPE | Median NMPJPE | Std |
|--------|-------------|---------------|-----|
| **MoveNet** | **11.5%** | **10.4%** | 5.5% |
| MediaPipe | 12.5% | 11.2% | 7.2% |
| YOLOv8-Pose | 12.9% | 11.3% | 6.8% |

**CSV-Quelle für Verifikation:** Die exakten Werte stammen aus dem `create_thesis_figures.py` / `generate_figures.py` Script, das die `analysis/results/` CSVs verarbeitet und Outlier-Filterung anwendet. Die `comprehensive_analysis.json` zeigt leicht andere Zahlen (MoveNet 12.7%) weil dort KEINE Outlier-Filterung angewendet wurde.

### B.3 All-Data Zahlen (VERALTET — docs/02 Aktualisierte Ergebnisse)

| Modell | NMPJPE | Std |
|--------|--------|-----|
| MoveNet | 14.9% | 11.3% |
| MediaPipe | 15.6% | 8.8% |
| YOLO | 19.2% | 12.9% |

**⚠ NICHT VERWENDEN.** Diese stammen aus der All-Data-Berechnung VOR Video-Kategorisierung und Outlier-Removal.

### B.4 Clean-Data OHNE Outlier-Removal (docs/04_RESULTS.md HAUPT-ERGEBNIS)

| Modell | NMPJPE | Std | Median |
|--------|--------|-----|--------|
| MoveNet | 12.7% | 5.7% | 9.8% |
| MediaPipe | 14.4% | 6.1% | 11.2% |
| YOLO | 17.0% | 8.4% | 11.2% |

**Quelle:** `docs/04_RESULTS.md`, Section "HAUPT-ERGEBNIS". Auch in `comprehensive_analysis.json` unter `model_benchmark.descriptive_statistics`.

### B.5 Statistische Signifikanz

| Vergleich | Test | p-Wert | Cohen's d | Signifikant? |
|-----------|------|--------|-----------|-------------|
| MediaPipe vs MoveNet | Wilcoxon signed-rank (per-video median) | 0.098 | 0.009 | Nein (p > 0.05) |
| MediaPipe vs YOLO | Wilcoxon signed-rank | < 0.001 | — | Ja |
| MoveNet vs YOLO | Wilcoxon signed-rank | < 0.001 | — | Ja |

**Quelle Thesis:** Table 5.1 Paragraph. `p = 0.098, Cohen's d = 0.009`.
**Quelle JSON:** `analysis/results/statistical_tests.json`, `significance.pairwise` — ABER diese nutzen t-Tests auf Frame-Level (nicht per-video Wilcoxon). Die Thesis-Zahlen (p=0.098) stammen aus einem separaten Wilcoxon-Test auf Video-Median-Werte.

### B.6 Rotation Robustness

| Modell | Frontal (0-20°) | Lateral (80-90°) | Degradation |
|--------|-----------------|-------------------|-------------|
| MediaPipe | 10.2% | 13.3% | +31% |
| MoveNet | 9.0% | 13.8% | +54% |
| YOLOv8-Pose | 9.4% | 14.9% | +58% |

**Quelle:** Thesis Table 5.3 (`05_results.tex`, Zeilen 117-122).
**Filter-Level:** Clean + Outlier Removed.
**CSV:** `analysis/results/rotation_analysis.csv` enthält ALLE Daten (inkl. Outlier), daher stimmen die Werte dort nicht direkt überein (z.B. MediaPipe 0-10 dort = 14.5% mean — das ist WITH outliers).

**ACHTUNG:** Die docs/02_PROBLEMS_AND_SOLUTIONS.md am Ende ("Aktualisierte Ergebnisse") zeigt alte c18-only Zahlen: MediaPipe +52%, MoveNet +27%, YOLO +46%. Diese verwenden nur c18-Daten und sind NICHT die kanonischen Thesis-Zahlen.

### B.7 Multi-Person Robustness

| Modell | Selection | Clean | Coach | Degradation |
|--------|-----------|-------|-------|-------------|
| MediaPipe | Torso size | 14.5% | 44.8% | +209% |
| MoveNet | BBox area | 14.8% | 64.9% | +340% |
| YOLOv8-Pose | BBox area | 17.7% | 66.1% | +274% |

**Quelle:** Thesis Table 5.4 (`05_results.tex`, Zeilen 151-155). Caption sagt: "all data, including outliers".
**Filter-Level:** 121 clean videos vs 5 coach videos, ALLE Frames inkl. Outlier (>100%).
**JSON:** `analysis/results/summary_statistics.json`, `coach_impact` — Werte dort: MediaPipe 14.506/44.813/208.9%, MoveNet 14.755/64.932/340.1%, YOLO 17.658/66.125/274.5%. Thesis-Werte sind gerundet.

**Diskrepanz-Hinweis:** `statistical_tests.json` zeigt unter `selection_robustness` andere Clean-Werte: MediaPipe 14.4%, MoveNet 12.7%, YOLO 17.0% mit Degradationen +215%, +390%, +289%. Diese nutzen offenbar outlier-removed clean-Daten. Die HANDOFF_PROMPT nutzt +209%/+340% (thesis-Version, inkl. Outlier).

### B.8 Temporal Stability (Jitter)

| Modell | Mean Jitter | Median Jitter |
|--------|-------------|---------------|
| MoveNet | 1.06% | 0.42% |
| YOLOv8-Pose | 1.12% | 0.38% |
| MediaPipe | 1.51% | 0.53% |

**Quelle:** Thesis Table 5.6 (`05_results.tex`, Zeilen 230-234).
**CSV:** `analysis/results/temporal_jitter.csv` — exakte Werte: MediaPipe 1.5128, MoveNet 1.0634, YOLO 1.1223.
**42% Differenz:** MediaPipe hat 42% höheren Jitter als MoveNet (1.51/1.06 - 1 ≈ 42.5%).

### B.9 Detection Completeness

| Modell | Full detection (12/12) | NMPJPE increase wenn <12 |
|--------|------------------------|--------------------------|
| YOLOv8-Pose | 87.8% | +15% |
| MoveNet | 79.2% | +27% |
| MediaPipe | 64.0% | +34% |

**Quelle:** Thesis Table 5.8 (`05_results.tex`, Zeilen 261-266).
**CSV:** `analysis/results/valid_joints_analysis.csv` — MediaPipe: 74,978 Frames mit 12/12 von 118,305 total (63.4%); MoveNet: 92,769 / 118,472 (78.3%); YOLO: 99,638 / 115,001 (86.6%). Die Thesis-Werte weichen leicht ab (64.0/79.2/87.8%) — vermutlich nach Outlier-Filterung berechnet.

### B.10 Inference Time

| Modell | Mean (ms) | Median (ms) | FPS | P95 (ms) | Std (ms) |
|--------|-----------|-------------|-----|----------|----------|
| MoveNet | 36.1 | 34.8 | 27.7 | 45.8 | 5.5 |
| YOLOv8-Pose | 52.9 | 51.1 | 18.9 | 60.3 | 8.3 |
| MediaPipe | 67.9 | 64.0 | 14.7 | 116.1 | 27.4 |

**Quelle:** Thesis Table 5.9 (`05_results.tex`, Zeilen 292-297).
**CSV:** `analysis/results/inference_benchmark.csv` — Werte stimmen exakt überein.
**Setup:** CPU-only, AMD Ryzen, 16GB RAM, 1500 Frames/Modell, 50-Frame Warmup.
**Script:** `benchmark_inference.py` (separate vom Hauptevaluations-Pipeline).

### B.11 Per-Joint Analysis

| Body Region | MediaPipe | MoveNet | YOLOv8 |
|-------------|-----------|---------|--------|
| Shoulders | 7–8% | 8–10% | 8–10% |
| Elbows | ~9% | ~10% | ~10% |
| Wrists | 9–10% | 11–12% | 12–13% |
| **Hips** | **16–17%** | 10–11% | 14–15% |
| Knees | 7–8% | ~6% | 7–8% |
| Ankles | ~12% | ~10% | ~10% |

**Quelle:** Thesis Table 5.10 (`05_results.tex`, Zeilen 326-333).
**CSV:** `analysis/results/per_joint_analysis.csv` — Exakte Werte (Mean). Beispiel: MediaPipe left_hip=19.23%, right_hip=17.74% → ~16-17% als Range. MoveNet left_hip=14.24%, right_hip=13.07% → ~10-14%.

**⚠ ACHTUNG:** Die per_joint_analysis.csv enthält Werte MIT Outliers. Die Thesis-Ranges (~16-17% für MediaPipe-Hips) scheinen aus Outlier-removed Daten berechnet zu sein, da die CSV höhere Rohwerte zeigt (19.2%/17.7%).

### B.12 Camera Comparison

| Modell | c17 raw | c18 raw | c17 filtered | c18 filtered |
|--------|---------|---------|--------------|--------------|
| MediaPipe | 15.2% | 13.6% | 11.8% | 13.1% |
| MoveNet | 14.1% | 11.4% | 9.6% | 11.2% |
| YOLOv8 | 20.4% | 13.9% | 11.3% | 13.7% |

**Quelle:** Thesis Table 5.2 (`05_results.tex`, Zeilen 72-82).
**JSON:** `analysis/results/summary_statistics.json`, `camera` Section — c17/c18 mean Werte stimmen überein (MediaPipe c17=15.46→15.2%, c18=13.61→13.6%).

### B.13 Person-Switch Frames

| Modell | c17 | c18 | Ratio |
|--------|-----|-----|-------|
| MediaPipe | 1.52% | 0.30% | 4.8× |
| MoveNet | 1.57% | 0.11% | 12.8× |
| YOLOv8 | 2.62% | 0.13% | 17.8× |

**Quelle:** Thesis Table 5.3 Person-Switch (`05_results.tex`, Zeilen 94-99).
**JSON:** `summary_statistics.json` → `camera.*.c17.pct_over_100` / `camera.*.c18.pct_over_100`. Exakt: MediaPipe c17=1.586%, c18=0.306%. Thesis rundet auf 1.52%/0.30% — leichte Diskrepanz, vermutlich weil Thesis nur Clean-Daten nutzt, JSON alle Daten.

### B.14 Gesamt-Statistiken

| Metrik | Wert | Quelle |
|--------|------|--------|
| Analysierte Frames | 363,529 | Thesis Sec 5.1 |
| Videos total | 126 | Thesis Table 3.1 |
| Clean Videos | 121 | Thesis Sec 3.1 |
| Coach Videos | 5 | Thesis Sec 3.1 |
| Patienten | 21 | Thesis Table 3.1 |
| Übungen | 6 | Thesis Table 3.1 |
| Kameras | 2 (c17 frontal, c18 lateral) | Thesis Table 3.1 |
| Frame-Step | 3 (10 Hz effektiv) | Thesis Sec 3.9 |
| Rotation frontal | 41.2% Frames | Thesis Sec 5.1 |
| Rotation intermediate | 17.6% Frames | Thesis Sec 5.1 |
| Rotation lateral | 41.3% Frames | Thesis Sec 5.1 |

---

## C. Confidence-Threshold-Inventar

### C.1 Modell-Detection-Thresholds (beim Inference)

| Datei | Zeile | Variable | Wert | Zweck | Final-Impact |
|-------|-------|----------|------|-------|-------------|
| `mediapipe_estimator.py` | 42 | `min_detection_confidence` | 0.1 | Minimale Person-Detection-Confidence | JA — bestimmt ob eine Person überhaupt detektiert wird |
| `mediapipe_estimator.py` | 87 | `min_tracking_confidence` | 0.5 | Tracking-Confidence (VIDEO-Mode) | NEIN — IMAGE-Mode wird verwendet, Tracking nicht aktiv |
| `mediapipe_estimator.py` | 88 | `num_poses` | 5 | Max. Anzahl detektierter Personen | JA — ermöglicht Multi-Person für Torso-Selection |
| `movenet_multipose_estimator.py` | 91 | `score > 0.1` (inline) | 0.1 | Person-Score-Filter bei Multi-Person-Selection | JA — Personen mit Score ≤ 0.1 werden ignoriert |
| `yolo_estimator.py` | 87 | `score > 0.3` (inline) | 0.3 | Person-Detection-Confidence bei Multi-Person | JA — Detections mit Score ≤ 0.3 werden ignoriert |

### C.2 Evaluation-Threshold (bei Metriken-Berechnung)

| Datei | Zeile | Variable | Wert | Zweck | Final-Impact |
|-------|-------|----------|------|-------|-------------|
| `evaluator.py` | 58 | `MIN_JOINT_CONFIDENCE` | 0.5 | Per-Joint Filter: Joints < 0.5 → NaN → aus NMPJPE-Berechnung ausgeschlossen | **JA — Haupteinfluss auf finale Zahlen.** Betrifft 6.8% MediaPipe-Joints, 2.2% MoveNet, ~0% YOLO |

### C.3 Outlier-Definitionen (keine Code-Thresholds, sondern Post-hoc)

| Definition | Schwelle | Wo angewendet | Impact |
|-----------|---------|---------------|--------|
| Extreme Outlier | NMPJPE > 100% | Thesis Accuracy Tabellen | Frames entfernt → primäre Accuracy-Zahlen |
| Wrong Person | Shoulder-Dist > 200px | `docs/01_METHODOLOGY.md` Sec. 5 (nicht im aktuellen Code) | Diagnostik, nicht in finaler Pipeline |
| Detection Failure | Alle Keypoints = (0,0) | Implizit in Evaluator | Frames excluded from NMPJPE |

### C.4 Welcher Threshold beeinflusst welche Thesis-Zahl?

- **Hauptzahlen (11.5%/12.5%/12.9%):** `MIN_JOINT_CONFIDENCE=0.5` + Outlier-Removal (>100%) + Clean-Data (121 Videos)
- **Multi-Person-Zahlen (+209%/+340%/+274%):** `MIN_JOINT_CONFIDENCE=0.5` angewendet, aber KEINE Outlier-Removal (>100% Frames einbezogen)
- **Jitter-Zahlen (1.06/1.12/1.51):** Unabhängig von Confidence-Threshold (Jitter = Frame-to-Frame Displacement)
- **Inference-Zahlen (27.7/18.9/14.7 FPS):** Kein Threshold relevant (reines Timing)

---

## D. Diskrepanzen und offene Fragen

### D.1 MediaPipe Default-Complexity: Code vs. Thesis

**Code:** `mediapipe_estimator.py` Zeile 42: `model_complexity: int = 2` (Default = Heavy)
**Thesis:** Table 3.2 sagt "Full (complexity=1)". Auch `docs/01_METHODOLOGY.md` sagt "Full (1)".
**Auflösung:** Der `__main__` Block in `inference/pipeline.py` (Zeile 340) nutzt `model_complexity=1`. Die Prediction-Dateien enthalten "MediaPipe_full" als Key. Der Code-Default ist also nicht der tatsächlich verwendete Wert. **Die Thesis ist korrekt.** Der Default-Wert im Konstruktor sollte auf 1 geändert werden, um Konsistenz herzustellen.

### D.2 YOLO Default Model Size: Code vs. Thesis

**Code:** `yolo_estimator.py` Zeile 18: `model_size: str = "m"` (Default = Medium)
**Thesis:** Table 3.2 sagt "Nano (n)".
**Auflösung:** Die tatsächliche Inference verwendete `model_size="n"`, sichtbar an den Prediction-Keys "YOLOv8-Pose_n" und der Inference-Benchmark CSV. Der Code-Default ist Medium, aber es wurde explizit Nano angegeben. **Die Thesis ist korrekt.**

### D.3 Drei verschiedene Clean-Accuracy-Zahlensets

| Quelle | MoveNet | MediaPipe | YOLO | Beschreibung |
|--------|---------|-----------|------|-------------|
| `docs/04_RESULTS.md` HAUPT-ERGEBNIS | 12.7% | 14.4% | 17.0% | Clean 121, MIT Outliers |
| Thesis Table 5.1 / HANDOFF_PROMPT | 11.5% | 12.5% | 12.9% | Clean 121, OHNE Outliers (>100% entfernt) |
| `docs/02` "Aktualisierte Ergebnisse" | 14.9% | 15.6% | 19.2% | All 126 Videos (VERALTET) |

**Problem:** `docs/04_RESULTS.md` wird als "kanonisch" bezeichnet, aber die Thesis verwendet eine zusätzliche Outlier-Filterung die in docs/04 nicht explizit dokumentiert ist. Die Outlier-Removal-Zahlen (11.5%/12.5%/12.9%) kommen aus den Figure-Generation-Scripts.

### D.4 Multi-Person-Degradation: Zwei Berechnungsweisen

| Quelle | MediaPipe | MoveNet | YOLO | Basis |
|--------|-----------|---------|------|-------|
| Thesis Table 5.4 | +209% | +340% | +274% | Clean=ALL frames, Coach=ALL frames |
| `statistical_tests.json` | +215% | +390% | +289% | Clean=outlier-removed, Coach=ALL |
| HANDOFF_PROMPT | +209% | +340% | — | Gleich wie Thesis |
| `docs/04_RESULTS.md` | +215% | +390% | +289% | Gleich wie statistical_tests |

**Problem:** docs/04_RESULTS.md und die Thesis verwenden verschiedene Baseline-Clean-Werte für die Degradations-Berechnung. Die Thesis nutzt Clean MIT Outliers als Baseline (14.5%/14.8%/17.7%), docs/04 nutzt Clean OHNE Outliers (14.4%/12.7%/17.0%). Dies erzeugt verschiedene Prozent-Degradationen.

### D.5 Per-Joint-Werte: Ranges vs. Exakte Zahlen

Die Thesis Table 5.10 zeigt per-joint NMPJPE als Ranges (z.B. "16–17%" für MediaPipe-Hips). Die `per_joint_analysis.csv` zeigt höhere Werte (left_hip=19.2%, right_hip=17.7%). Die Thesis-Ranges entsprechen vermutlich outlier-removed Werten, aber dies ist nicht explizit dokumentiert.

### D.6 Wilcoxon-Test: p=0.098 nicht aus den CSV-Daten reproduzierbar

Die Thesis berichtet p=0.098 (Wilcoxon signed-rank auf per-video Median NMPJPE). Die `statistical_tests.json` enthält Frame-Level-t-Tests (p=7.17e-61 für MediaPipe vs MoveNet). Der per-video Wilcoxon wurde vermutlich in einem separaten Script berechnet, das nicht in `analysis/results/` abgelegt ist.

### D.7 Person-Switch-Raten: Leichte Abweichungen

`summary_statistics.json` zeigt MediaPipe c17 = 1.586% over 100%. Die Thesis sagt 1.52%. Vermutlich weil summary_statistics alle 126 Videos enthält, die Thesis aber nur die 121 Clean-Videos auswertet.

### D.8 Coach-Video PM_011: Nicht in Segmentation.csv als Extra-Person markiert

`data/Segmentation (1).csv` zeigt für PM_011-Ex3 `extra_person_in_cam17=0` bei ALLEN Repetitionen. Trotzdem wurde PM_011-c17 als Coach-Video klassifiziert (38.0% / 59.7% / 66.2% NMPJPE). Mögliche Erklärungen:
- Coach erscheint zwischen Repetitionen
- Segmentation-Annotationen unvollständig
- Klassifikation erfolgte durch Modell-Fehleranalyse, nicht durch Segmentation.csv

### D.9 `docs/00_PROJECT_OVERVIEW.md` veraltet

Sagt MediaPipe "Heavy (complexity=2)" — tatsächlich wurde Full (complexity=1) verwendet. Sagt "~185.000 Frames" — tatsächlich 363,529 (weil frame_step=3, also ~121K pro Modell × 3 Modelle). Die Forschungsfrage fokussiert auf Rotation, aber die Thesis hat sich zu einer "holistischen Evaluation" erweitert.

---

## E. Multi-Person-Analyse Detail

### E.1 Die 5 Coach-Videos

Alle sind c17-Videos, alle Exercise 3:

| Video | MediaPipe | MoveNet | YOLO | Segmentation-Annotation |
|-------|-----------|---------|------|------------------------|
| PM_010-c17 | 81.9% | 70.7% | 69.9% | extra_person=2/3 in Reps 1+5 |
| PM_011-c17 | 38.0% | 59.7% | 66.2% | **extra_person=0 in allen Reps** |
| PM_108-c17 | 17.2% | 46.2% | 51.1% | extra_person=1-3 in Reps 1-2, 6-10 |
| PM_119-c17 | 24.1% | 74.0% | 73.0% | extra_person=3 in Reps 1-4, 11 |
| PM_121-c17 | 30.9% | 60.3% | 75.3% | extra_person=1-3 in Reps 1-4, 6-11 |

**Quelle NMPJPE:** Thesis Table 5.5 / `docs/04_RESULTS.md` Coach-Detail-Tabelle.

### E.2 Segmentation.csv: Extra-Person-Kodierung

Das `extra_person_in_cam17` Feld hat folgende Werte-Verteilung (über alle 1072 Repetitionen):

| Wert | Häufigkeit | Interpretation (vermutlich) |
|------|-----------|---------------------------|
| 0 | 686 | Keine extra Person |
| 1 | 117 | Person weit entfernt / kaum sichtbar |
| 2 | 206 | Person teilweise sichtbar |
| 3 | 63 | Person deutlich sichtbar |

### E.3 Pervasivität des Multi-Person-Problems

**62 von 63 c17-Videos** (=63 Patient-Exercise-Combos) haben mindestens eine Repetition mit `extra_person_in_cam17 > 0`. Das bedeutet fast JEDES c17-Video hat irgendwann eine zweite Person im Bild.

Für `extra_person_in_cam18` ist das Problem viel seltener:
- PM_000-Ex1, PM_010-Ex3, PM_018-Ex4, PM_020-Ex4, PM_022-Ex6, PM_024-Ex1, PM_025-Ex2, PM_029-Ex6, PM_126-Ex6 — nur 9 Videos mit extra Person in c18.

### E.4 Sporadische Multi-Person-Frames jenseits der 5 Coach-Videos

Die docs/04_RESULTS.md erwähnt "~26 weitere Videos mit sporadischen Multi-Person-Frames". Dies wird bestätigt durch:
- c17 hat 5-18× mehr Frames >100% NMPJPE als c18 (Table 5.3)
- 62 c17-Videos haben extra_person > 0 laut Segmentation.csv
- Differenz: Die 5 als "Coach" klassifizierten Videos sind nur die extremen Fälle; die restlichen ~57 Videos haben AUCH sporadische Extra-Personen, die zu gelegentlichen Person-Switches führen

### E.5 Torso-Size-Paradox

In 156 von 186 Disagreement-Frames (84%) hatte der Coach einen GRÖSSEREN Torso als der Patient, aber MediaPipe selektierte trotzdem den Patienten. Thesis Table 5.5 zeigt:

| Video | Coach-Torso (px) | Patient-Torso (px) | Ratio |
|-------|-----------------|-------------------|-------|
| PM_108-c17 | 469 | 199 | 2.36× |
| PM_119-c17 | 441 | 222 | 1.98× |
| PM_121-c17 | 445 | 221 | 2.02× |
| PM_011-c17 | 230 | 253 | 0.91× |

**Hypothese:** MediaPipe's Face/Upper-Body-Detector (erste Pipeline-Stufe) erkennt Personen von hinten nicht → Coach wird gar nicht erst detektiert → Torso-Selection nur unter den detektierten Personen (= nur Patient).

**Gegenprobe (PM_010):** Wenn Coach frontal zur Kamera steht, wird er detektiert → alle Modelle versagen.

---

## F. Dataset-Kompatibilitäts-Profil

### F.1 Was die Pipeline als Input erwartet

| Komponente | Format | Details |
|-----------|--------|---------|
| **Video** | MP4 (oder AVI mit CV2-kompatiblem Codec) | Beliebige Auflösung, 30 FPS bevorzugt |
| **Video-Namenskonvention** | `PM_XXX-Camera{17,18}-30fps*.mp4` | Regex: `(PM_\d+)-Camera(\d+)-30fps` in `data_loader.py` Zeile 81 |
| **GT 2D** | NumPy `.npy`, Shape `(num_frames, 26, 2)` | 26 Joints in MoCap-Hierarchie-Reihenfolge, Pixel-Koordinaten |
| **GT 3D** | NumPy `.npy`, Shape `(num_frames, 26, 4)` | 26 Joints, 4 Werte pro Joint (x, y, z + ?) |
| **Verzeichnisstruktur** | `data/videos/Ex{N}/`, `data/gt_2d/Ex{N}/`, `data/gt_3d/Ex{N}/` | Exercise-basierte Unterordner |

### F.2 Keypoint-Mapping-Anforderung

Ein neues Dataset muss 12 Joints liefern die auf COCO-Indices 5-16 mappbar sind:

| Required Joint | COCO-Index | Aktuelles GT-Mapping (REHAB24-6) |
|---------------|------------|----------------------------------|
| Left Shoulder | 5 | GT Joint 7 (LeftArm) |
| Right Shoulder | 6 | GT Joint 12 (RightArm) |
| Left Elbow | 7 | GT Joint 8 (LeftForeArm) |
| Right Elbow | 8 | GT Joint 13 (RightForeArm) |
| Left Wrist | 9 | GT Joint 9 (LeftHand) |
| Right Wrist | 10 | GT Joint 14 (RightHand) |
| Left Hip | 11 | GT Joint 16 (LeftUpLeg) |
| Right Hip | 12 | GT Joint 21 (RightUpLeg) |
| Left Knee | 13 | GT Joint 17 (LeftLeg) |
| Right Knee | 14 | GT Joint 22 (RightLeg) |
| Left Ankle | 15 | GT Joint 18 (LeftFoot) |
| Right Ankle | 16 | GT Joint 23 (RightFoot) |

**Code:** `src/pose_evaluation/data/keypoint_mapping.py` — `COCO_TO_GT_MAPPING` Dict.

### F.3 Was angepasst werden müsste für einen neuen Datensatz

| Komponente | Aufwand | Was zu tun ist |
|-----------|---------|---------------|
| **data_loader.py** | Mittel | Neue `_parse_video()` Methode für die Namenskonvention des neuen Datasets |
| **keypoint_mapping.py** | Gering | Neues `GT_TO_COCO_MAPPING` für das Joint-Format des neuen Datasets |
| **pipeline.py** | Mittel | `calculate_rotation_angle()` benötigt ggf. neuen `C17_FRONTAL_OFFSET` oder entfällt wenn Kameras kalibriert sind |
| **evaluator.py** | Gering | `get_gt_2d_path()` anpassen; `calculate_torso_length_gt()` wenn GT-Indices anders |
| **Estimatoren** | Keine | Die 3 Modelle sind dataset-unabhängig |
| **Inference** | Keine | `frame_step` ggf. anpassen an FPS |

### F.4 Was funktioniert direkt (ohne Änderung)

- Alle 3 Estimatoren (MediaPipe, MoveNet, YOLO)
- NMPJPE/PCK Metrik-Berechnung (auf COCO-Format standardisiert)
- Torso-Length Normalisierung (wenn Schulter/Hüfte Joints vorhanden)
- Jitter-Berechnung
- Confidence-Filtering

### F.5 Anforderungen für Cross-Dataset-Vergleich (Lars' Vorschlag)

Per HANDOFF_PROMPT Section 3.1:
- Dataset muss COCO-Format Keypoints haben (oder mappbar)
- Öffentlich verfügbar
- Rehabilitation oder klinischer Kontext
- Video-Format mit Ground Truth

Wenn ein zweiter Datensatz hinzugefügt wird:
- **Neue Thesis-Section:** "Cross-Dataset Generalization" in Kap. 5
- **Nur Accuracy-Vergleich (NMPJPE)** — kein Jitter/Rotation (da kamera-spezifisch)
- Alle bestehenden REHAB24-6-Ergebnisse bleiben unverändert

---

## G. Ergänzung: Offene Betreuer-Aufgaben (Status-Referenz)

Aus HANDOFF_PROMPT Section 8:

| Aufgabe | Status | Betrifft |
|---------|--------|----------|
| Text de-LLMizen | TODO | Alle Kapitel |
| Redundante Table/Figure-Paare | TODO | Kap. 5 (Multi-Person, Temporal, Detection) |
| Model Selection als Designentscheidung | TODO | Kap. 3 Sec 3.6 |
| Qualitative Beispiele | TODO | Kap. 5 |
| Titel in cfg.tex | TODO | cfg.tex |
| Modellvarianten | AUSSTEHEND (Lars) | Kap. 3+5 |
| Detection Threshold Studie | AUSSTEHEND (Lars) | Kap. 3+5 |
| Zweiter Datensatz | AUSSTEHEND (Lars) | Kap. 3+5+6 |
| Statistische Analyse mit Abhängigkeiten | AUSSTEHEND (Lars) | Kap. 3+5 |
| Frame-by-Frame Rechtfertigung | TODO (1 Satz) | Kap. 3 oder 4 |
