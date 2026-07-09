# Präsentations-Masterplan v2 — „The Ranking That Refused to Exist"

> Ersetzt v1 (nach der v1-Storyline wurde defense_slides.pptx gebaut; Delta unten).
> 30-Min-Slot, Ziel 28:00 · Englisch · 22 Folien + 9 Backups (B1–B9 unverändert).

## Strategische Kernentscheidung

**Erkenntnis-Drama statt Ergebnisbericht.** Erwartung aufbauen (ein Gewinner) → Bruch
(letzte zwei Dimensionen) → Verschärfung (die naheliegende Erklärung scheitert an den
Daten — Dreisatz 6.3) → Auflösung auf höherer Ebene (drei Profile + Framework).
Neutralisiert „stumpfer Vergleich / nicht komplex genug" strukturell.

**Leitmotiv:** Folie 5 (Paradigmen) und Folie 19 (Profile) haben dieselbe
Drei-Spalten-Geometrie → „Bauprinzipien am Anfang, in den Messungen wiedergefunden am
Ende" — sichtbar, ohne es sagen zu müssen.

**Fortschritts-Marker:** Ergebnis-Folien tragen „Dimension k of 6 · Name" — ersetzt
die Landkarten-Folie, macht den Bruch sichtbar (4 of 6 → plötzlich anderer Gewinner).

## Dichte-Regel (Pitch → Defense umlernen)

Defense-Folie = Bühnenbild, 60–120 Sek: (1) Assertion-Titel (2 Sek erfassbar),
(2) EIN dominantes visuelles Element, (3) max. 3–5 Textanker, beim Sprechen
nacheinander aktiviert (hinzeigen, Zahl laut lesen, kommentieren). Nie zwei
konkurrierende Strukturen auf einer Folie.

## Opener (auswendig)

"Physiotherapy works when exercises are done correctly and regularly — but most of it
happens at home, unsupervised, and adherence is poor. A smartphone camera with a pose
estimation model could close that gap. My thesis asks: which of today's mobile models
can you actually trust in a living room? And the answer turned out to be more
interesting than a ranking."

## Ablauf (Akt · Folie · Zeit · Kern/Choreografie)

### AKT I — Problem & Kandidaten (0:00–6:45)
1. **Titel** · 0:30 · Opener.
2. **Home rehabilitation fails without feedback** · 1:15 · 3 Anker (Adhärenz →
   Telerehab → Kamera als Sensor). Zügig.
3. **Standard benchmarks answer a different question** · 2:00 · 4 Constraints (2×2).
   Versprechen: "each becomes a measured dimension — you'll see all six."
4. **Five research questions** · 0:30 · zeigen + Merkformel, nicht vorlesen.
5. **Three candidates, three construction principles** · 2:30 · GALL-MOMENT 1.
   Mechanik präzise (detect→crop→estimate / full-image→centers / detector+pose-head).
   Proaktive Begründung: Auswahlkriterien. Geometrie = Folie 19.

### AKT II — Messinstrument (6:45–11:45)
6. **REHAB24-6: proxy benchmark** · 2:00 · Skelett-Frame; „proxy, not clinical
   validation"; auf Therapeutin im Hintergrund zeigen (Foreshadowing: "keep her in mind").
7. **One fixed protocol isolates the models** · 2:15 · Pipeline + NMPJPE-Einzeiler;
   auf der Folie nur „we evaluate models, not pipelines" — Rest in Notes (ENTLASTET ggü. v1).
8. **Frames lie about sample size — honest statistics** · 1:00 · 10 Cluster, exakter
   1024-Welten-Test, Holm, Bootstrap in 3 Sätzen. KEIN Dimensionen-Grid mehr. Details B1.
   → SIGNPOST: "That's the instrument. Six dimensions, nine configurations, one
   protocol. Here's what we found."

### AKT III — Erkenntnisreise (11:45–23:45)
**Phase A — scheinbarer Durchmarsch:**
9. **1 of 6 · Accuracy: MoveNet — in all ten clusters** · 2:00 · Boxplot führen
   ("each dot is one person"); Deep-Point p = 2/1024.
10. **2 of 6 · Viewpoint: everyone degrades, MoveNet least** · 0:45 · 3 Zahlen.
    [RUBBER — live kürzbar]
11. **3 of 6 · Stability: MoveNet family sweeps top three** · 1:00 · 3.81/5.22/6.03;
    "raw outputs, true motion cancels."
12. **4 of 6 · Speed: MoveNet leads the mains** · 1:15 · Scatter; Varianten in 1 Satz.
    → SIGNPOST (Wendepunkt, Pause davor): "Four dimensions in, MoveNet sweeps. A
    simple story. The remaining two dimensions break it."

**Phase B — der Bruch:**
13. **5 of 6 · Completeness: YOLOv8 delivers the full skeleton** · 0:45 · erster Riss.
14. **6 of 6 · Multi-person: harmless on average — until it's the coach** · 1:00 ·
    Zwei-Regime-Split.
15. **One frame, three behaviors** · 1:30 · 4-Panel führen; MoveNet-Box auf Therapeut
    = "a selection failure, not a detection failure."
16. **The obvious explanation fails — in three steps** · 2:00 · HÖHEPUNKT, langsamste
    Folie. Dreisatz konsistent→bricht→bestätigt; "the mechanism sits deeper, and we
    leave it open." [UNANTASTBAR]

**Phase C — das reifere Bild:**
17. **Failure signatures mirror the architectures** · 1:30 · GALL-MOMENT 2. Drei
    Signaturen + Confound-Satz LAUT: "one family per paradigm — observational, not causal."
18. **Rankings don't travel: the COCO check** · 1:00 · Slopegraph; "limited
    transferability, not benchmark invalidity." [RUBBER]

### AKT IV — Synthese & Klammer (23:45–28:00)
19. **Three architectural profiles — not one ranking** · 1:45 · SPIEGELFOLIE zu 5.
    Kernfolie. [UNANTASTBAR] Einstieg mit Brücken-Satz zur Kernfrage: "So back to
    the original question — which model for home rehabilitation?" (hörbare Rückkehr
    zum Anwendungsthema — kein Architektur-Exkurs ohne Payoff).
20. **A decision guide, not a verdict** · 1:00 · Default MoveNet; wann YOLO/MP/
    Lightning; "deployment guidance matters as much as model choice." [RUBBER]
21. **What this evidence can and cannot say** · 1:00 · 4 Limitation→Next-Step-Paare;
    "most important: real patients."
22. **Conclusion + Danke** · 1:00 · 2 Befunde + Closer.

## Closer (auswendig)

"So — can you trust a mobile pose estimator in a living room? Yes. But which one
depends on what your application cannot afford to lose. MoveNet MultiPose is the
default; the three architectures fail in three different ways, and knowing those ways
is as valuable as any ranking. The next step toward the clinic is validation on real
patients. Thank you."

## Live-Steuerung

- **Rubber-Sections** (je −20–30 Sek möglich): 10, 18, 20.
- **Unantastbar/dehnbar:** 16 (Dreisatz), 19 (Profile).
- Regel: Zeitprobleme hinten lösen (18/20), NIE am Höhepunkt.

## Gall-Momente (wo Präzision demonstriert wird)

Folie 5 (Architektur-Mechanik) · Folie 8 (exakter Test) · Folie 17 (Confound selbst
benennen) · Folie 21 (Grenzen proaktiv). Vier proaktive Begründungen im Vortrag:
Proxy-Framing (6), frame-independent (7), Cluster-Statistik (8), Modellwahl (5).
Alles andere: Backups + Q&A-Katalog.

## Delta zur gebauten defense_slides.pptx (v1)

1. Speed-Folie: Position 16 → 12 (Sweep-Phase; ermöglicht Wendepunkt-Signpost).
2. Fortschritts-Marker „k of 6" auf Ergebnis-Folien; Dimensionen-Grid (alte Folie 8) raus.
3. Folie 7 + 2 entlasten (Begründungen → Notes).
4. Signposts + Opener/Closer wörtlich in die Sprechnotizen.
5. Folie 19 geometrisch exakt auf Folie 5 spiegeln.
→ Umsetzung: build_slides.py anpassen, regenerieren (~15 Min). Befehl: „umsetzen".
