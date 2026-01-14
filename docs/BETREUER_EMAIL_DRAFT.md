# Email-Draft an Betreuer

---

**Betreff:** Bachelorarbeit - Zwischenstand Pose Estimation Evaluation & Thesis-Richtung

---

Sehr geehrter Herr [Name],

ich möchte Ihnen einen Zwischenstand meiner Bachelorarbeit geben und Ihre Einschätzung zur geplanten Richtung einholen.

## 1. Durchgeführte Evaluation

Ich habe eine umfassende Evaluation von drei mobilen Pose-Estimation-Modellen (MediaPipe, MoveNet, YOLOv8-Pose) auf dem REHAB24-6 Dataset durchgeführt:

- **Datenbasis:** 363.529 Frames aus 126 Videos (21 Patienten, 6 Übungen, 2 Kameras)
- **Ground Truth:** Motion Capture (optische Marker)
- **Metrik:** NMPJPE (Normalized Mean Per Joint Position Error)

## 2. Zentrale Ergebnisse

| Dimension | Ergebnis |
|-----------|----------|
| **Accuracy** | MediaPipe ≈ MoveNet (statistisch nicht unterscheidbar, p=0.098) |
| **Rotation Robustheit** | MediaPipe am besten (+31% vs +54% Fehleranstieg bei seitlicher Ansicht) |
| **Multi-Person Robustheit** | MediaPipe 2x besser (Torso-Selection vs BBox-Selection) |
| **Temporal Stability** | MoveNet am stabilsten (42% weniger Frame-zu-Frame Jitter) |

**Haupterkenntnis:** Kein Modell dominiert. Für realistische Home-Based Rehabilitation (suboptimale Kamerawinkel, gelegentlich andere Personen im Bild) ist MediaPipe robuster. In kontrollierten Umgebungen ist MoveNet minimal besser.

## 3. Literaturvergleich

Ich habe einen systematischen Literaturvergleich durchgeführt (6 relevante Paper). Die wichtigsten Gaps, die unsere Arbeit füllt:

| Aspekt | Stand der Literatur | Unsere Contribution |
|--------|---------------------|---------------------|
| MoveNet auf Reha-Daten | Nicht evaluiert | Erste Evaluation mit MoCap Ground Truth |
| Selection-Strategien | Nicht verglichen | Torso vs BBox quantifiziert |
| Temporal Stability | Nicht systematisch untersucht | Erste Jitter-Analyse für mobile HPE |
| Rotation + Multi-Person kombiniert | Einzeln untersucht, nicht zusammen | Kombinierte Analyse |

## 4. Geplante Thesis-Richtung

Ich plane einen breiten Evaluations-Fokus statt eines einzelnen Findings:

**Arbeitstitel:** "Evaluating Mobile Pose Estimation Models for Home-Based Rehabilitation: Accuracy, Stability, and Robustness"

**Research Questions:**
1. Wie genau sind mobile HPE-Modelle auf echten Reha-Daten?
2. Wie stabil sind die Predictions über Zeit (Temporal Stability)?
3. Wie robust sind die Modelle bei suboptimalen Bedingungen (Rotation, Multi-Person)?
4. Welche praktischen Empfehlungen ergeben sich für mobile Reha-Apps?

## 5. Frage an Sie

Ist dieser breite Evaluations-Fokus aus Ihrer Sicht tragfähig für eine Bachelorarbeit, oder sollte ich stärker auf einen einzelnen Aspekt fokussieren?

Im Anhang finden Sie die vollständige Ergebnisübersicht als PDF.

Mit freundlichen Grüßen,
[Name]

---

## Anhang für Email

**Mitschicken:**
- `analysis/evaluation_results.pdf` (aus .tex kompiliert)
- Optional: `analysis/EVALUATION_SUMMARY.md` als Markdown

---

## Notizen (nicht in Email)

**Mögliche Rückfragen des Betreuers:**
- "Warum diese drei Modelle?" → Mobile-optimiert, relevant für Previa Health
- "N=5 für Multi-Person?" → Limitation, aber Effekt sehr klar
- "Was ist neu vs. UCO Paper?" → MoveNet, Jitter-Analyse, Selection-Vergleich
