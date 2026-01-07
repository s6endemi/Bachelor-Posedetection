# ZWISCHENSTAND: Pose Estimation Vergleich

> **Letzte Aktualisierung:** 07.01.2026 (Session 4)
> **Status:** Kamera-Offset + MediaPipe-Confidence GELOEST + Evaluator mit Filter
> **Fortschritt:** ~75%

---

## Quick Status

| Phase | Status | Details |
|-------|--------|---------|
| 1. Grundlagen | ✅ | Dataset, Keypoint-Mapping |
| 2. Estimators | ✅ | 3 Modelle implementiert |
| 3. Selection | ✅ | Multi-Person Handling (modell-spezifisch!) |
| 4. Pipeline | ✅ | Inference + Rotation |
| 5. Evaluation | ✅ | Metriken + Aggregation + Visualisierungen |
| 6. Mini-Test | ✅ | 500 Frames validiert |
| 7. **Ex1 Inference** | 🔄 | **9/26 Videos fertig, morgen fortsetzen** |
| 8. **Kamera-Offset** | ✅ | **GELOEST: C17_FRONTAL_OFFSET = 65 Grad** |
| 9. Full-Run | ⬜ | Nach Ex1 fertig |
| 10. Analyse | ⬜ | Plots, Statistik |
| 11. Thesis | ⬜ | Schreiben |

---

## ✅ GELOEST: Kamera-Koordinatensystem

### Das Problem (war)
Die berechneten Rotationswinkel waren relativ zum **MoCap-Koordinatensystem**, NICHT zur Kamera.

### Loesung (implementiert)
**Empirische Offset-Bestimmung** aus Videos PM_114, PM_122, PM_109:

```python
# In pipeline.py
C17_FRONTAL_OFFSET = 65.0  # MoCap-Winkel bei dem Person frontal zu c17 steht

def mocap_to_camera_relative(mocap_angle, camera):
    c17_relative = abs(mocap_angle - C17_FRONTAL_OFFSET)
    if camera == 'c17':
        return c17_relative
    else:  # c18 ist 90 Grad zu c17 gedreht
        return 90.0 - c17_relative
```

### Validierung
| Video | MoCap | c17 (berechnet) | c18 (berechnet) | Beobachtung |
|-------|-------|-----------------|-----------------|-------------|
| PM_000 | ~50 | 15 (frontal) | 75 (seitlich) | Passt! |
| PM_002 | ~2 | 63 (seitlich) | 27 (frontal) | Passt! |

### Erste Ergebnisse (9 Videos, kamera-relativ)
| Winkel | NMPJPE (alle Modelle) |
|--------|----------------------|
| 0-20 (frontal) | ~8-11% |
| 70-90 (seitlich) | ~15-18% |

**Hypothese bestaetigt: Seitliche Ansichten haben ~2x hoehere Fehler!**

---

## Aktuelle Konfiguration

| Modell | Variante | Selection | Config |
|--------|----------|-----------|--------|
| MediaPipe | Heavy | Torso-Größe | `confidence=0.1` |
| MoveNet | **MultiPose** | BBox Area | Lightning |
| YOLO | Nano/Medium | BBox Area | Default |

**Evaluation:** `MIN_JOINT_CONFIDENCE = 0.5` - Joints mit niedriger Confidence werden bei NMPJPE-Berechnung ausgeschlossen.

---

## Mini-Test Ergebnisse (500 Frames)

| Modell | Fehler | NMPJPE (0-10°) | NMPJPE (50-60°) |
|--------|--------|----------------|-----------------|
| MediaPipe | 0.4% | 10.4% | 19.6% |
| MoveNet MP | 0.0% | ~14% | ~14% |
| YOLO | 0.0% | 10.4% | 14.3% |

**→ Vollständige Ergebnisse nach Full-Run in `docs/04_RESULTS.md`**

---

## Naechste Schritte

### MORGEN: Ex1 Inference fortsetzen
Die Pipeline wurde angepasst und speichert jetzt kamera-relative Winkel.
Bestehende 9 .npz Dateien wurden bereits transformiert.

```bash
# Ex1 fortsetzen (startet automatisch bei Video 10)
.venv/Scripts/python run_inference.py --exercise Ex1
```

### DANACH
1. Full-Run (alle 126 Videos)
2. Finale Evaluation mit `Evaluator`
3. Statistische Analyse + Plots
4. Thesis schreiben

---

## Detaillierte Dokumentation

| Dokument | Inhalt | Wann updaten? |
|----------|--------|---------------|
| [`docs/00_PROJECT_OVERVIEW.md`](docs/00_PROJECT_OVERVIEW.md) | Forschungsfrage, Modelle, Dataset | Bei Scope-Änderung |
| [`docs/01_METHODOLOGY.md`](docs/01_METHODOLOGY.md) | Metriken, Rotation, Selection-Strategien | Bei Methodik-Änderung |
| [`docs/02_PROBLEMS_AND_SOLUTIONS.md`](docs/02_PROBLEMS_AND_SOLUTIONS.md) | **Die Journey!** Bugs, Fixes, Learnings | Bei neuen Erkenntnissen |
| [`docs/03_EXPERIMENTS.md`](docs/03_EXPERIMENTS.md) | Alle Tests, Setup, Ergebnisse | Nach jedem Experiment |
| [`docs/04_RESULTS.md`](docs/04_RESULTS.md) | Finale Ergebnisse (nach Full-Run) | Nach Full-Run |
| [`docs/05_THESIS_OUTLINE.md`](docs/05_THESIS_OUTLINE.md) | Thesis-Gliederung (~40 Seiten) | Bei Struktur-Änderung |

---

## Wichtigste Erkenntnisse (Kurzfassung)

### 1. Selection-Strategien sind modell-spezifisch
```
YOLO/MoveNet: BBox Area (echte Boxen vom Detektor)
MediaPipe:    Torso-Größe (keine echten Boxen!)
```
**→ Details in `docs/01_METHODOLOGY.md` Abschnitt 4**

### 2. Gefundene & behobene Probleme
| Problem | Ursache | Lösung |
|---------|---------|--------|
| MediaPipe 29% Failures | confidence=0.5 | → 0.1 |
| YOLO 37% NMPJPE | Erste Person statt größte | → BBox Selection |
| MoveNet falsche Person | SinglePose Limitation | → MultiPose |

**→ Details in `docs/02_PROBLEMS_AND_SOLUTIONS.md`**

### 3. Korrelation ≠ Kausalität
"Rotations-Fehler bei 50-60°" war eigentlich Person-Verwechslung!

**→ Details in `docs/02_PROBLEMS_AND_SOLUTIONS.md` Problem 2**

---

## Update-Checkliste

Bei Änderungen diese Docs prüfen:

- [ ] **Neuer Bug gefunden?** → `02_PROBLEMS_AND_SOLUTIONS.md`
- [ ] **Experiment durchgeführt?** → `03_EXPERIMENTS.md`
- [ ] **Full-Run fertig?** → `04_RESULTS.md` + `ZWISCHENSTAND.md`
- [ ] **Methodik geändert?** → `01_METHODOLOGY.md`
- [ ] **Thesis-Struktur geändert?** → `05_THESIS_OUTLINE.md`

---

## Dateien & Pfade

### Code
```
run_inference.py              # Hauptskript
src/pose_evaluation/          # Gesamter Code
├── estimators/               # 3 Modelle
├── inference/                # Pipeline
└── evaluation/               # Metriken
```

### Daten
```
data/videos/                  # 126 Videos
data/ground_truth/            # Motion Capture GT
data/predictions/             # Output (.npz)
```

### Dokumentation
```
ZWISCHENSTAND.md              # ← Diese Datei (Status)
docs/
├── 00_PROJECT_OVERVIEW.md    # Was & Warum
├── 01_METHODOLOGY.md         # Wie
├── 02_PROBLEMS_AND_SOLUTIONS.md  # Journey
├── 03_EXPERIMENTS.md         # Tests
├── 04_RESULTS.md             # Ergebnisse
├── 05_THESIS_OUTLINE.md      # Thesis-Plan
└── archive/                  # Alte Versionen
```

---

## Für Previa Health

| Empfehlung | Details |
|------------|---------|
| **Modell** | YOLO - stabilste Performance |
| **MediaPipe Config** | `confidence=0.1` |
| **MoveNet** | MultiPose, nicht SinglePose |
| **Multi-Person** | Immer größte Person wählen |

### Offene Fragen (nach Full-Run)
- Ab welchem Winkel User-Warnung?
- Kritischer Winkel θ_crit?
- Welche Joints leiden am meisten?

---

## Erstellte Visualisierungen (diese Session)

```
evaluation_nmpjpe_explained.png   # NMPJPE vs Rotation + Was bedeutet 10%?
comparison_rotation_0_10.png      # Vergleich bei 0-10° (MoCap)
comparison_rotation_20_30.png     # Vergleich bei 20-30°
comparison_rotation_40_50.png     # Vergleich bei 40-50°
comparison_rotation_50_60.png     # Vergleich bei 50-60°
comparison_rotation_60_70.png     # Vergleich bei 60-70°
debug_rotation_check.png          # Debug: PM_002 vs PM_000 Winkelvergleich
```

---

## Changelog

| Datum | Änderung | Betroffene Docs |
|-------|----------|-----------------|
| 07.01 | Evaluator: Confidence-Filter 0.5 implementiert | evaluator.py |
| 06.01 | **Kamera-Koordinatensystem Problem entdeckt** | 02, ZWISCHENSTAND |
| 06.01 | Visualisierungen erstellt | - |
| 06.01 | Ex1 Inference gestartet (8/26 fertig) | - |
| 06.01 | MoCap Rotation Variation dokumentiert (~2.5° Std) | 02 |
| 06.01 | Docs-Struktur erstellt | Alle |
| 06.01 | MediaPipe BBox getestet (schlecht) | 02, 03 |
| 05.01 | MoveNet MultiPose implementiert | 01, 02 |
| 05.01 | YOLO BBox Selection | 01, 02 |
| 05.01 | MediaPipe confidence=0.1 | 01, 02 |
