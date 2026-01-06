# Thesis Outline: Gliederung der Bachelorarbeit

**Arbeitstitel:**
> "Einfluss des Körper-Rotationswinkels auf die Genauigkeit von 2D Pose Estimation: Ein systematischer Vergleich von MediaPipe, MoveNet und YOLOv8-Pose"

---

## Gliederung

### 1. Einleitung (ca. 3-4 Seiten)

#### 1.1 Motivation
- Pose Estimation in mobilen Gesundheitsanwendungen
- Praktisches Problem: Nutzer stehen nicht immer frontal
- Beispiel: Previa Health und Bewegungsanalyse

#### 1.2 Problemstellung
- COCO Benchmark hauptsächlich frontale Ansichten
- Fehlende systematische Rotation-Analyse
- Keine praktischen Guidelines für Entwickler

#### 1.3 Forschungsfrage
> Wie verändert sich die 2D-Keypoint-Genauigkeit von Pose Estimation Modellen in Abhängigkeit vom Rotationswinkel der Person zur Kamera?

#### 1.4 Beitrag der Arbeit
1. Systematisches Evaluationsprotokoll
2. Quantitative Daten zur Degradation
3. Praktische Guidelines
4. Architekturvergleich

#### 1.5 Aufbau der Arbeit
Kurze Kapitelübersicht

---

### 2. Grundlagen und Related Work (ca. 8-10 Seiten)

#### 2.1 Pose Estimation Grundlagen
- Definition und Anwendungsgebiete
- Keypoint-basierte vs. Mesh-basierte Ansätze
- 2D vs. 3D Pose Estimation

#### 2.2 Architekturen
- **Top-Down:** Erst Detection, dann Keypoints (MediaPipe)
- **Bottom-Up:** Erst Keypoints, dann Gruppierung (MoveNet)
- **One-Stage:** Alles in einem Schritt (YOLO)

#### 2.3 Modelle im Detail
- **MediaPipe Pose (BlazePose)**
  - Architektur: Two-Stage
  - 33 Keypoints inkl. Gesicht/Hände
  - Pseudo-3D Output

- **MoveNet**
  - Architektur: Heatmap-basiert
  - 17 COCO Keypoints
  - Varianten: SinglePose vs MultiPose

- **YOLOv8-Pose**
  - Architektur: Anchor-free One-Stage
  - 17 COCO Keypoints
  - Varianten nach Größe (n, s, m, l, x)

#### 2.4 Evaluation von Pose Estimation
- Metriken: PCK, OKS, MPJPE, NMPJPE
- Benchmarks: COCO, MPII
- Limitationen bestehender Benchmarks

#### 2.5 Viewpoint-Variation
- Bestehende Arbeiten zu Multi-View
- Self-Occlusion bei Rotation
- Forschungslücke identifizieren

---

### 3. Methodik (ca. 8-10 Seiten)

#### 3.1 Dataset
- REHAB24-6: Beschreibung, Statistiken
- Ground Truth: Motion Capture System
- Begründung der Wahl

#### 3.2 Modellauswahl
- Kriterien: Mobile-tauglich, verschiedene Architekturen
- Varianten-Entscheidung (Heavy, MultiPose, etc.)

#### 3.3 Rotationswinkel-Berechnung
- Formel: arctan2(|dz|, |dx|) aus 3D GT
- Warum aus Ground Truth (Zirkelschluss vermeiden)
- Binning-Strategie (10°-Bins)

#### 3.4 Keypoint-Mapping
- Problem: Verschiedene Formate
- Lösung: 12 gemeinsame Keypoints
- Mapping-Tabelle

#### 3.5 Fehlermetrik
- NMPJPE: Definition und Begründung
- Normalisierung durch Torso-Länge
- Interpretation der Werte

#### 3.6 Person-Selection **(Wichtiger Abschnitt!)**
- Problem: Multi-Person Szenarien
- Modell-spezifische Lösungen:
  - YOLO/MoveNet: BBox Area
  - MediaPipe: Torso-Größe
- Begründung: Echte BBox vs. Pseudo-BBox

#### 3.7 Ausreißer-Handling
- Definitionen: Detection Failure, Wrong Person, High Error
- Ausschluss-Kriterien
- Transparenz in der Berichterstattung

#### 3.8 Experimentelles Design
- Unabhängige Variable: Rotationswinkel
- Abhängige Variablen: NMPJPE, Per-Joint Error
- Kontrollierte Faktoren

---

### 4. Implementierung (ca. 4-5 Seiten)

#### 4.1 Software-Architektur
- Abstrakte Basisklasse `PoseEstimator`
- Modularer Aufbau
- Pipeline-Flow

#### 4.2 Technische Details
- Modell-Konfigurationen
- Confidence-Thresholds
- Input-Preprocessing

#### 4.3 Datenverarbeitung
- Frame-Extraktion
- Parallele Inference
- Speicherformat (.npz)

#### 4.4 Reproduzierbarkeit
- Code-Repository
- Abhängigkeiten
- Ausführungsanleitung

---

### 5. Ergebnisse (ca. 8-10 Seiten)

#### 5.1 Deskriptive Statistik
- Anzahl Frames pro Winkel-Bin
- Detection Performance pro Modell
- Ausreißer-Statistik

#### 5.2 NMPJPE vs Rotationswinkel
- Hauptergebnis: Tabelle + Grafik
- Vergleich der drei Modelle
- Konfidenzintervalle

#### 5.3 Per-Joint Analyse
- Welche Joints sind anfällig?
- Heatmap-Visualisierung
- Unterschiede zwischen Modellen

#### 5.4 Statistische Tests
- Signifikanz der Unterschiede
- ANOVA + Post-hoc Tests
- Effektstärken

#### 5.5 Qualitative Beispiele
- Visualisierung bei 0°, 45°, 90°
- Typische Fehlerquellen

---

### 6. Diskussion (ca. 6-8 Seiten)

#### 6.1 Interpretation der Ergebnisse
- Beantwortung der Forschungsfragen
- Überprüfung der Hypothesen
- Erklärung der Muster

#### 6.2 Architektur-Vergleich
- Top-Down vs Bottom-Up vs One-Stage bei Rotation
- Stärken und Schwächen
- Empfehlungen

#### 6.3 Praktische Implikationen
- Kritischer Winkel θ_crit
- Guidelines für App-Entwickler
- Empfehlungen für Previa Health

#### 6.4 Lessons Learned **(Wichtiger Abschnitt!)**
- MediaPipe Confidence Problem
- Person-Selection Herausforderungen
- SinglePose vs MultiPose
- Warum BBox nicht immer funktioniert

#### 6.5 Limitationen
- Dataset-spezifisch (Rehabilitation)
- Kontrollierte Umgebung
- Single-Person Fokus
- Statische Winkelbetrachtung

#### 6.6 Ausblick
- Dynamische Winkeländerung
- Weitere Modelle
- Multi-Person Evaluation
- Outdoor-Szenarien

---

### 7. Fazit (ca. 1-2 Seiten)

- Zusammenfassung der Hauptergebnisse
- Beitrag zur Forschung
- Praktischer Nutzen
- Abschlussstatement

---

### Anhang

#### A. Keypoint-Mapping Tabellen
#### B. Vollständige Ergebnistabellen
#### C. Zusätzliche Visualisierungen
#### D. Code-Auszüge
#### E. Statistische Details

---

## Seitenverteilung (geschätzt)

| Kapitel | Seiten |
|---------|--------|
| Einleitung | 3-4 |
| Grundlagen | 8-10 |
| Methodik | 8-10 |
| Implementierung | 4-5 |
| Ergebnisse | 8-10 |
| Diskussion | 6-8 |
| Fazit | 1-2 |
| **Gesamt** | **38-49** |

+ Anhang, Verzeichnisse, Literatur

---

## Wichtige Punkte für die Thesis

### 1. Selection-Strategien hervorheben
Die modell-spezifischen Selection-Strategien sind ein **eigener methodischer Beitrag**:
> "Die Person-Selektion wurde modell-spezifisch implementiert..."

### 2. Lessons Learned einbauen
Die Discovery-Journey ist wertvoll für Discussion:
- Confidence-Problem
- Korrelation ≠ Kausalität
- Architektur-Limitationen

### 3. Praktische Relevanz betonen
Verbindung zu Previa Health und realen Anwendungen:
- Konkrete Schwellenwerte
- Implementierbare Guidelines

### 4. Ehrliche Limitationen
- Dataset aus spezifischem Kontext
- Nicht alle Edge Cases abgedeckt
- Raum für weitere Forschung

---

## Literatur (Vorläufig)

### Modelle
- Bazarevsky et al. (2020). BlazePose
- Google (2021). MoveNet
- Jocher et al. (2023). Ultralytics YOLOv8

### Benchmarks
- Lin et al. (2014). Microsoft COCO
- Andriluka et al. (2014). MPII Human Pose

### Metriken
- Sigal et al. (2012). Human Pose Estimation

### Dataset
- REHAB24-6. Zenodo.

---

## Zeitplan

| Phase | Status |
|-------|--------|
| Grundlagen & Related Work | ⬜ |
| Methodik (kann jetzt geschrieben werden) | ⬜ |
| Implementierung (kann jetzt geschrieben werden) | ⬜ |
| Full-Run durchführen | ⬜ |
| Ergebnisse analysieren | ⬜ |
| Ergebnisse schreiben | ⬜ |
| Diskussion schreiben | ⬜ |
| Einleitung + Fazit | ⬜ |
| Review & Korrektur | ⬜ |
