# ZWISCHENSTAND: Pose Estimation Vergleich

> **Letzte Aktualisierung:** 06.01.2026 (Session 2)
> **Status:** Ex1 Inference läuft, KRITISCHES PROBLEM entdeckt: Kamera-Koordinatensystem
> **Fortschritt:** ~55%

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
| 7. **Ex1 Inference** | 🔄 | **8/26 Videos fertig, läuft im Hintergrund** |
| 8. **Kamera-Offset** | ⚠️ | **KRITISCH: Rotation ist MoCap-relativ, nicht Kamera-relativ!** |
| 9. Full-Run | ⬜ | Nach Kamera-Offset-Fix |
| 10. Analyse | ⬜ | Plots, Statistik |
| 11. Thesis | ⬜ | Schreiben |

---

## ⚠️ KRITISCHES PROBLEM: Rotationswinkel-Koordinatensystem

### Das Problem
Die berechneten Rotationswinkel sind relativ zum **MoCap-Koordinatensystem**, NICHT zur Kamera!

```
Beobachtung:
- PM_002 hat 0-10° Rotation → Person steht SEITLICH zur Kamera
- PM_000 hat 45-55° Rotation → Person steht FRONTALER zur Kamera

Das ist genau UMGEKEHRT zu dem was man erwarten würde!
```

### Ursache
Die Kameras stehen nicht senkrecht zum MoCap-Koordinatensystem:
- **Camera17**: Person erscheint ~frontal wenn MoCap-Winkel ~45-50°
- **Camera18**: Person erscheint ~seitlich (90° gedreht zu Camera17)

### Lösung (TODO)
**Option 1: PnP (Perspective-n-Point)**
- Wir haben 3D GT + 2D GT → können Kamera-Extrinsics berechnen
- Sauberer, mathematisch korrekter Ansatz
- Ermöglicht exakte Transformation MoCap→Kamera

**Option 2: Empirische Schätzung**
- Aus Videos visuell den Offset bestimmen
- Schneller, aber weniger genau

### Nächster Schritt
PnP mit `cv2.solvePnP()` implementieren um Kamera-Rotation zu berechnen

---

## Aktuelle Konfiguration

| Modell | Variante | Selection | Config |
|--------|----------|-----------|--------|
| MediaPipe | Heavy | Torso-Größe | `confidence=0.1` |
| MoveNet | **MultiPose** | BBox Area | Lightning |
| YOLO | Nano/Medium | BBox Area | Default |

---

## Mini-Test Ergebnisse (500 Frames)

| Modell | Fehler | NMPJPE (0-10°) | NMPJPE (50-60°) |
|--------|--------|----------------|-----------------|
| MediaPipe | 0.4% | 10.4% | 19.6% |
| MoveNet MP | 0.0% | ~14% | ~14% |
| YOLO | 0.0% | 10.4% | 14.3% |

**→ Vollständige Ergebnisse nach Full-Run in `docs/04_RESULTS.md`**

---

## Nächste Schritte

### JETZT: Full-Run starten
```bash
python run_inference.py                  # Alle 126 Videos (~2-4h)
python run_inference.py --exercise Ex1   # Nur Ex1 (~20min)
```

### DANACH
1. Evaluation mit `Evaluator`
2. NMPJPE pro Winkel-Bin
3. Plots erstellen
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
| 06.01 | **Kamera-Koordinatensystem Problem entdeckt** | 02, ZWISCHENSTAND |
| 06.01 | Visualisierungen erstellt | - |
| 06.01 | Ex1 Inference gestartet (8/26 fertig) | - |
| 06.01 | MoCap Rotation Variation dokumentiert (~2.5° Std) | 02 |
| 06.01 | Docs-Struktur erstellt | Alle |
| 06.01 | MediaPipe BBox getestet (schlecht) | 02, 03 |
| 05.01 | MoveNet MultiPose implementiert | 01, 02 |
| 05.01 | YOLO BBox Selection | 01, 02 |
| 05.01 | MediaPipe confidence=0.1 | 01, 02 |
