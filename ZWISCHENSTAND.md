# ZWISCHENSTAND: Pose Estimation Vergleich

> **Letzte Aktualisierung:** 09.01.2026 (Session 7 - Video-Kategorisierung abgeschlossen)
> **Status:** Clean-Data Analyse ABGESCHLOSSEN
> **Fortschritt:** ~75%

---

## Quick Status

| Phase | Status | Details |
|-------|--------|---------|
| 1. Grundlagen | DONE | Dataset, Keypoint-Mapping |
| 2. Estimators | DONE | 3 Modelle implementiert |
| 3. Selection | DONE | Multi-Person Handling (modell-spezifisch!) |
| 4. Pipeline | DONE | Inference + Rotation + Frame-Stepping |
| 5. Full-Run | DONE | 126 Videos, 122.400 Frames |
| 6. **Neu-Evaluation** | DONE | **Saubere Metriken, korrigierte Zahlen** |
| 7. **Video-Kategorisierung** | DONE | **121 Clean, 5 Coach-Interaction** |
| 8. Statistische Analyse | PENDING | ANOVA, Regression |
| 9. Visualisierungen | PENDING | Plots fuer Thesis |
| 10. Thesis | PENDING | Schreiben |

---

## WICHTIGSTE ERKENNTNISSE (Session 7)

### 1. Alte Dokumentation war FEHLERHAFT

Die vorherigen Zahlen (54.9% YOLO c17 Fehler) waren falsch - vermutlich Bug im alten Evaluator (frame_step nicht beruecksichtigt).

| Metrik | Alte Doku (FALSCH) | Neu-Evaluation (KORREKT) |
|--------|-------------------|--------------------------|
| YOLO c17 | 54.9% | **24.6%** |
| YOLO c18 | 15.3% | **13.9%** |
| Videos >30% (YOLO c17) | 49.2% | **25.4%** |

### 2. Clean-Data Modell-Ranking (121 Videos)

| Modell | NMPJPE | Std | Bewertung |
|--------|--------|-----|-----------|
| **MoveNet** | **13.0%** | 5.7% | BESTE WAHL |
| MediaPipe | 14.6% | 6.1% | Gut, stabil |
| YOLO | 17.2% | 8.4% | c17-Problem |

### 3. Clean-Data c17 vs c18 (ohne Coach)

| Modell | c17 | c18 | Differenz | Interpretation |
|--------|-----|-----|-----------|----------------|
| MediaPipe | 15.9% | 13.4% | +2.5% | Minimal |
| MoveNet | 14.6% | 11.5% | +3.1% | Moderat |
| **YOLO** | **20.9%** | **13.9%** | **+7.1%** | **Signifikant** |

### 4. Selection-Robustheit bei Coach-Videos

| Modell | Clean c17 | Coach c17 | Anstieg | Selection |
|--------|-----------|-----------|---------|-----------|
| MediaPipe | 15.9% | 38.4% | **+22.5%** | Torso-Groesse |
| MoveNet | 14.6% | 62.2% | **+47.6%** | BBox + Score |
| YOLO | 20.9% | 67.1% | **+46.2%** | BBox + Score |

**Kernaussage:** MediaPipe's Torso-Selection ist **2x robuster** als BBox-Selection.

### 5. Rotations-Effekt (auf sauberen c18-Daten)

| Rotation | MediaPipe | MoveNet | YOLO |
|----------|-----------|---------|------|
| 30-40 (schraeg) | 10.2% | 9.7% | 10.8% |
| 70-90 (seitlich) | 15.5% | 12.3% | 15.8% |
| **Anstieg** | **+52%** | **+27%** | **+46%** |

MoveNet zeigt geringsten Rotations-Effekt!

---

## Strategischer Plan

### Phase 1: Daten-Integritaet (DONE)
- [x] YOLO Score-Filter implementiert (macht keinen grossen Unterschied)
- [x] Saubere Neu-Evaluation durchgefuehrt
- [x] Alte fehlerhafte Zahlen identifiziert

### Phase 2: Analyse-Framework (DONE)
- [x] run_evaluation.py erstellt (reproduzierbar)
- [x] Videos kategorisiert (121 Clean, 5 Coach)
- [x] Separate Analysen durchgefuehrt

### Phase 3: Akademische Tiefe (PENDING)
- [ ] Statistische Signifikanz (ANOVA, Tukey HSD)
- [ ] Regression: Fehler-Modell als Funktion der Rotation
- [ ] Literaturrecherche und -vergleich
- [ ] Per-Joint Analyse vertiefen

### Phase 4: Outputs (PENDING)
- [ ] Publication-Quality Visualisierungen
- [ ] Thesis schreiben

---

## Aktuelle Konfiguration

| Modell | Variante | Selection | Robustheit |
|--------|----------|-----------|------------|
| MediaPipe | Full (1) | Torso-Groesse | **Robust** bei Coach |
| MoveNet | MultiPose Lightning | BBox + Score>0.1 | Mittel |
| YOLO | Nano (n) | BBox + Score>0.3 | **Anfaellig** bei Coach |

---

## Daten-Ueberblick

| Metrik | Wert |
|--------|------|
| Total Videos | 126 |
| Total Frames (evaluiert) | ~363.000 |
| Frames pro Video | ~970 (durchschnitt) |
| Rotation Range | 0-90 Grad |
| Comparable Joints | 12 |
| Videos mit Coach-Problem | 5 (alle c17) |

---

## Dateien und Pfade

### Wichtige Dateien
```
data/predictions/Ex1-Ex6/*.npz   # 126 Prediction Files
data/evaluation_results.csv       # Neu-Evaluation Ergebnisse
run_evaluation.py                 # Sauberes Evaluation-Script
docs/                             # Dokumentation (wird aktualisiert)
```

### Commands
```bash
# Neu-Evaluation ausfuehren
.venv/Scripts/python run_evaluation.py

# Virtual Environment
.venv/Scripts/python <script.py>
```

---

## Fuer Previa Health (aktualisiert)

| Empfehlung | Details |
|------------|---------|
| **Modell** | MoveNet MultiPose - beste Performance + guter Rotations-Robustheit |
| **Alternative** | MediaPipe - robuster bei Multi-Person-Szenarien |
| **YOLO** | Nur wenn Single-Person garantiert |
| **Multi-Person** | Warnung ausgeben, Neupositionierung empfehlen |
| **Seitliche Ansicht** | Fehler steigt um 27-52%, vermeiden wenn moeglich |

---

## Naechste Schritte

1. ~~Videos kategorisieren (Clean vs Coach)~~ DONE
2. ~~Separate Analysen durchfuehren~~ DONE
3. Statistische Tests (ANOVA, Tukey HSD)
4. Visualisierungen erstellen
5. Thesis schreiben

---

## Changelog

| Datum | Aenderung |
|-------|-----------|
| 09.01 | **Video-Kategorisierung abgeschlossen**: 121 Clean, 5 Coach |
| 09.01 | Clean-Data Ergebnisse: MoveNet 13.0%, MediaPipe 14.6%, YOLO 17.2% |
| 09.01 | Selection-Robustheit: MediaPipe 2x robuster bei Multi-Person |
| 09.01 | **Session 7** - Saubere Neu-Evaluation, alte Zahlen korrigiert |
| 08.01 | Session 6 - Deep Evaluation (Zahlen waren fehlerhaft) |
| 07.01 | Full-Run abgeschlossen |
