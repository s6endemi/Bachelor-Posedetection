# Chapter 5 — Results (Verteidigungsanalyse)

> Quelle: `thesis/chapters/05_results.tex` (final, vollständig gelesen)
> Alle Zahlen → `key_numbers.md` (Spickzettel). Hier: Struktur, Storys, Fallen.

## 1. Die Ergebnis-Story in 6 Dimensionen (+ Zusatzanalysen)

1. **Accuracy (5.2):** MoveNet 10,52% < MediaPipe 12,53% < YOLOv8 12,77% (frame-level
   mean, valid subset). Ordnung bleibt nach Cluster-Aggregation. Alle drei
   Paarvergleiche nach Holm signifikant. MoveNet gewinnt **alle 6 Übungen** (Appendix).
2. **Varianten (5.3):** MoveNet_MultiPose bleibt #1 selbst gegen größere Familien-
   varianten (YOLOv8_Medium 10,86 knapp dahinter). ⚠️ Überraschung: **MultiPose
   schlägt SP_Thunder deutlich** (10,52 vs. 13,33).
3. **Cross-Dataset (5.4):** REHAB-#1 MoveNet_MultiPose ist auf COCO nur #4; COCO-#1
   ist YOLOv8_Medium. MediaPipe stürzt auf COCO ab (Full: Rang 5→8, 17,21%).
   MoveNet-SP-Varianten steigen auf COCO (Thunder 7→3). YOLOv8 am stabilsten.
   Bei den drei Hauptmodellen: MoveNet auf beiden #1, MediaPipe/YOLO tauschen.
4. **Viewpoint (5.5):** frontal→lateral Degradation: MoveNet +17,8% < MediaPipe
   +27,5% < YOLOv8 +31,7% (relative Steigerung des frontalen Fehlers).
5. **Multi-Person (5.6):** Segmentation-Subset (severity ≥2): kaum Effekt (MP +1,2%,
   MoveNet −5,0%, YOLO −4,4%) → keine große Degradation im breiten Subset. Coach-
   Subset (5 Videos, 1.178 sampled frames): das eigentliche Stress-Szenario —
   2+-Personen-Frequenz 23/16/62% (MP/MoveNet/YOLO), Outlier-Raten 9,1/13,8/13,9%,
   Coach-Mean-NMPJPE (VOR Outlier-Cut!) 45,4/62,2/66,0%. Per-Video: PM_010 bricht
   ALLE drei (>65%); in den anderen 4 bleibt MediaPipe <40%.
6. **Temporal Stability (5.7):** MoveNet 3,81% < YOLOv8 5,22% < MediaPipe 6,03%.
   ⚠️ Reihenfolge ≠ Accuracy-Reihenfolge! Alle 9: **MoveNet-Familie belegt Top 3**
   (Thunder 3,31, Lightning 3,61, MultiPose 3,81).
7. **Detection Completeness (5.8):** YOLOv8 79,1% volle 12/12-Detektionen > MediaPipe
   55,1% > MoveNet 38,2%. (Wieder andere Reihenfolge → drei Profile!)
8. **Inference CPU (5.9):** SP_Lightning 100,6 FPS; MultiPose 27,7; MediaPipe_Full
   14,7; YOLOv8n 18,9; langsamste YOLOv8_Medium 4,6. MediaPipe hat hohe P95-Ausreißer
   (Heavy 304 ms). Speed-Accuracy-Frontier: Familien-Accuracy-Gewinne kosten viel
   Latenz (MP Heavy, YOLO Medium). Hinweis im Text: Throughput = native Stacks, kein
   backend-normalisierter Architektur-Benchmark.
9. **Per-Joint (5.10):** MediaPipe gewinnt NUR Schultern (8,76). MoveNet gewinnt alle
   anderen Regionen. **MediaPipe-Hips 16,72 vs. MoveNet 11,28** → Hip-Ausschluss:
   MP 12,53→11,66, Gap zu MoveNet 2,01pp→1,21pp (Basis der Hip-Hypothese in Kap. 6).
10. **Failure-Modes (5.11):** 4er-Taxonomie, in Reihenfolge angewandt:
    Missing-Detection → Confidence-Collapse (<6/12 Joints nach Filter) →
    Multi-Person-Confusion (Outlier + severity≥2) → Keypoint-Displacement (Outlier
    ohne Zweitperson). Signaturen: MediaPipe 69,6% Keypoint-Displacement (von 1.583);
    MoveNet 51,4% Confidence-Collapse (von 1.875); YOLOv8 63,8% Missing-Detection
    (von 5.610; 3.580 Frames ganz ohne Prediction ≈ 20× MediaPipe-Rate).
    Multi-Person-Confusion überall ≤11%. Robust bei Cutoff-Variation 4–8.

## 2. Zahlen-Fallen (wo Prüfer Diskrepanzen sehen könnten)

### Falle A: Frame-Gap 2,01pp vs. Cluster-Differenz 1,01pp
Tab 5.1: 12,53−10,52 = 2,01pp (frame-level means). Tab 5.2: MediaPipe−MoveNet nur
1,01pp (cluster-level). **Warum verschieden?** Cluster-Kette nutzt MEDIANE pro Sequenz
→ MediaPipes hohe Streuung (Std 7,05 vs. MoveNet 4,66; schwere Fehler-Frames) bläht
den Frame-MEAN auf, die median-basierte Aggregation dämpft das.
> EN: "The frame-level mean weights every frame, so MediaPipe's heavy-tailed error
> distribution inflates it; the cluster pipeline aggregates per-sequence medians,
> which damps those tails. Both levels agree on the ordering — that's what matters."

### Falle B: YOLO-Median (11,23) < MediaPipe-Median (11,26), aber Mean umgekehrt
Median-Kreuzung! Heißt: Im typischen Frame sind YOLO und MediaPipe praktisch gleich;
MediaPipe hat mehr masse in mittleren Fehlern, YOLO dafür… tatsächlich sind beide
fast identisch — der Mean-Unterschied (12,53 vs 12,77) ist klein und die Verteilungen
überlappen stark (Tab 5.2: p=0,0352, knapp). Nicht überinterpretieren: "MediaPipe and
YOLOv8 are close; the significant but small difference favors MediaPipe."

### Falle C: p = 0,0020 zweimal — das IST 2/1024
MediaPipe−MoveNet und MoveNet−YOLO haben beide p=0,0020 = 2/1024 = der minimal
mögliche zweiseitige p-Wert → **MoveNet gewinnt in ALLEN 10 Clustern** gegen beide.
Diese Verbindung aktiv nennen können — zeigt echtes Test-Verständnis!

### Falle D: "Multi-Person macht MoveNet BESSER (−5,0%)?!"
Nein — kein kausaler Effekt. Das Segmentation-Subset ist nicht kontrolliert
zusammengesetzt (andere Übungen/Zeitpunkte/Viewpoints als Komplement). Aussage nur:
**keine große Degradation im breiten Subset**; der harte Test ist der Coach-Fall.
Kap. 5 interpretiert bewusst nicht (Kapitel-Rollen!).

### Falle E: MultiPose schlägt SP_Thunder (10,52 vs. 13,33)
Überraschend (Thunder gilt als die genauere SP-Variante). Plausible Erklärung (Q&A,
als Hypothese): SP-Modelle sind für zentrierte, formatfüllende Einzelpersonen designt;
frame-independent ohne intelligentes Cropping sehen sie das volle Bild → Person klein
im Input (192²/256²) → Auflösungsverlust. MultiPose arbeitet nativ auf dem vollen
Bild mit Kandidaten. SP_Lightning zusätzlich: nur 94.395 valide Frames (viel Filter-
Verlust). ⚠️ Prüfen, ob Kap. 6 das aufgreift — sonst als eigene Hypothese kennzeichnen.

### Falle F: Coach-Mean 45,4% etc. — andere Datenbasis!
Coach-NMPJPE wird VOR dem >100%-Outlier-Cut berechnet (Caption Tab 5.7!), damit
katastrophale Fehler sichtbar bleiben. Nicht mit valid-subset-Zahlen (10,52 etc.)
vergleichen — anderes Regime (failure behavior vs. central tendency).

## 3. Auflösung der offenen Checks aus ch03

- ✅ **367.200** = 126 Sequenzen × 3 Hauptmodelle = frame-level observations
  (122.400 gesampelte Frames × 3). Kette: 367.200 → 363.074 (ohne Detection-
  Failures) → 354.879 (valid central-tendency subset).
- ✅ **63 von 65 (exakt geklärt, 9.7.2026):** Die 65 publizierten Recordings enthalten
  eine in ZWEI Teile gesplittete Ex5-Session — **PM_117a und PM_117b** (a/b-Suffixe in
  allen Dateien: Videos, 2D-/3D-GT). Diese fallen aus dem Standard-Identifier-Schema
  (PM_XXX-CameraYY); evaluiert wurden die 63 regulären Recording-IDs, jede mit beiden
  Views + GT → 126 Sequenzen. Nichts mit vollständigen Standard-Daten wurde verworfen.

## 3b. Verständnis-Klärungen (Session 7.7.2026)

- **354.879 / „valid central-tendency subset" — der Trichter:** 367.200 (122.400
  Frames × 3 Modelle) → −Detection-Failures = 363.074 → −Outlier >100% Torso u.
  MoCap-Fehler = 354.879. „Central tendency" = Statistikbegriff für Mittel/Median →
  „das Subset, auf dem typische Fehler berechnet werden". Aussortiertes wandert in
  die Failure-Analysen. Filter greifen PRO Modell → verschiedene n (115.711–119.729).
- **Tab 5.6 (Multi-Person-Subset):** Frames mit severity≥2 vs. Rest — fast kein
  Unterschied (+1,2/−5,0/−4,4%). Bedeutung: bloße Anwesenheit einer Zweitperson ist
  harmlos = erste Hälfte der Zwei-Regime-Erkenntnis; das Risiko sitzt im Coach-Fall
  (anhaltend prominente Person → katastrophale Switches). ⚠️ Negative Vorzeichen
  NICHT kausal deuten (Subset nicht kontrolliert zusammengesetzt).
- **Fig 5.1 (Boxplot) lesen:** Punkt = eine Person (Cluster-NMPJPE = Mittel ihrer
  Sequenz-Mediane); Box = mittlere 50% der 10 Personen. Aussage: MoveNets GESAMTE
  Verteilung liegt unter den anderen = konsistenter Vorsprung, nicht ausreißer-
  getrieben; MP/YOLO überlappen. Visuelle Begründung für p = 2/1024.
- **Fig 5.4 (Coach-Vierpanel, PM_119-c17 Frame 105):** MediaPipe (grün) nur Patientin;
  MoveNet (rot) detektiert beide, largest-area wählt THERAPEUTEN (Selektions-, nicht
  Detektionsfehler!); YOLO (blau) detektiert 3, wählt Patientin. Kernpunkt:
  Multi-Person-Versagen ist kein einzelner Mechanismus — Detektion, Selektion oder
  gar nicht, je Modell anders. Caption „illustrative" — Statistik in Tab 5.7.

## 3c. Mündliches Skript (EN, pro Sektion 1–3 Sätze)

- **5.1:** "126 sequences, 367,000 model-frame observations, 355,000 valid after
  filtering; viewpoints are strongly bimodal — ~41% frontal, ~42% lateral."
- **5.2:** "MoveNet is most accurate at 10.5% NMPJPE vs. 12.5 and 12.8. All pairwise
  differences survive Holm — MoveNet wins in all ten clusters, the smallest possible
  p-value of the exact test. First on all six exercises."
- **5.3:** "MultiPose stays first even against larger variants; families show the
  expected accuracy-speed ladders — exception: MoveNet SinglePose falls behind."
- **5.4:** "Rankings don't fully transfer: REHAB winner → rank 4 on COCO; YOLOv8
  Medium leads there; MediaPipe drops sharply. Main models: MoveNet first on both,
  MediaPipe/YOLO swap."
- **5.5:** "All degrade frontal→lateral — MoveNet least (+18%), YOLOv8 most (+32%)."
- **5.6:** "Dataset-wide multi-person subset: barely any change — mere presence of a
  second person is harmless. The risk appears in the coach extreme case — that's why
  the thesis splits multi-person behavior into two regimes."
- **5.7:** "MoveNet lowest displacement at 3.8% of torso length, MediaPipe highest at
  6 — the MoveNet family occupies the top three."
- **5.8:** "YOLOv8 returns the complete skeleton most often — 79% vs. 55 and 38."
- **5.9:** "SP Lightning hits 100 FPS on CPU; mains run ~28/19/15. Family accuracy
  gains cost latency — Heavy and Medium drop to 6 and 5 FPS."
- **5.10:** "MediaPipe wins only shoulders; pronounced hip problem — 16.7 vs. 11.3.
  Excluding hips shrinks the gap from 2.0 to 1.2 points."
- **5.11:** "Distinct failure signatures: MediaPipe — confident misprediction;
  MoveNet — confidence collapse, graceful degradation; YOLOv8 — missing detections,
  largest overall failure count."

## 4. Folien-Kandidaten aus Kapitel 5

- **Accuracy-Folie:** Boxplot (fig_accuracy_boxplot) + die drei Zahlen + "significant
  after Holm; MoveNet wins in all 10 clusters".
- **Drei-Profile-Vorbereitung:** eine Folie mit den drei Reihenfolgen nebeneinander
  (Accuracy: MoveNet | Stability: MoveNet | Completeness: YOLO | Coach: MediaPipe) —
  das ist DIE Kernbotschaft.
- **Coach-Folie:** coach_detection_comparison.png (4-Panel, PM_119 Frame 105) —
  visuell stärkste Folie der Arbeit. + die 23/16/62 vs. 9,1/13,8/13,9-Zahlen.
- **Speed-Accuracy-Scatter** (fig_speed_accuracy_scatter): Pflicht für RQ5/Deployment.
- **Failure-Taxonomie:** gestapelte Balken (failure_mode_taxonomy) — die 3 Signaturen.
- Rotation als Backup-Folie (Tab 5.5 reicht im Vortrag als ein Satz).
