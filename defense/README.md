# Defense Preparation — Master Plan

> Verteidigung: ~8. Juli 2026 · 30 Min Vortrag (Englisch, PowerPoint) + Fragerunde
> Hauptfragesteller: voraussichtlich **Prof. Jürgen Gall** (HPE-Experte, hohes/strenges CV-Niveau)
> Erstbetreuer: Lars Doorenbos

## Single Source of Truth

**Nur `thesis/chapters/*.tex` + `thesis/abstract.tex` (finale, abgegebene Arbeit) zählen.**
Alle anderen Dokumente (HANDOFF_*.md, docs/, thesis_status_report*, analysis/) sind
historisch und enthalten teilweise **veraltete Zahlen** — niemals als Faktenquelle verwenden.
Kanonische Zahlenbasis der finalen Arbeit: `evaluation_v2/results/`.

## Strategie gegen die "nicht komplex genug"-Sorge

Die Arbeit gewinnt nicht durch Themenkomplexität, sondern durch **methodische
Souveränität**. Im Vortrag und Q&A sichtbar machen, wo die versteckte Schwierigkeit steckt:

1. MoCap-zu-Pixel-Ground-Truth (Projektion, Koordinatensysteme)
2. Skelett-Definition-Mismatch zwischen Modellen und GT (Keypoint-Mapping, Hip-Offset-Hypothese)
3. Person-Selection bei Multi-Person (keine Strategie ist perfekt — PM_010-Gegenprobe)
4. Statistik mit Abhängigkeiten (Subject-Cluster, gepaarte Permutationstests, Holm, Bootstrap)
5. Faire Protokoll-Isolation (frame-independent: Modelle, nicht Pipelines evaluieren)
6. Drei architektonische Profile statt eines Rankings — deployment-relevante Synthese

Regel für Q&A: **Jede Modell-Frage eine Ebene tiefer beantworten können, als die Arbeit
es verlangt** (Architektur-Interna, Trainingsdaten, Loss-Design, Metrik-Definitionen).

## Dateien

| Datei | Inhalt |
|---|---|
| `ch01_introduction.md` … `ch07_conclusion.md` | Pro Kapitel: Kernaussagen, Memorier-Fakten, Design-Entscheidungen + Verteidigung, Prüferfragen mit EN-Musterantworten, Folien-Kandidaten |
| `key_numbers.md` | Kernzahlen-Spickzettel (entsteht bei Kapitel 5) |
| `fundamentals_drill.md` | CV/HPE-Grundlagenwissen für Gall (entsteht bei Kapitel 2) |
| `qa_catalog.md` | Kumulativer Fragenkatalog mit Musterantworten (EN) |
| `slides_storyline.md` | Folien-Plan, gespeist aus den "Folien-Kandidaten"-Sektionen |

## Lernmethode (aktiv, nicht passiv)

1. Claude liest Kapitel vollständig → erstellt Kapitel-MD → Besprechung im Chat.
2. Eren liest Kapitel + MD selbst nach.
3. **Aktiver Recall:** Zu Beginn jeder neuen Sitzung stellt Claude 5 Drill-Fragen zum
   vorherigen Kapitel (auf Englisch, wie in der Verteidigung).
4. Q&A-Katalog wächst kumulativ; am Ende Mock-Defense mit Timing.

## Fortschritt

- [x] Kapitel 1 Introduction — analysiert (ch01_introduction.md)
- [ ] Kapitel 2 Background — **Gall-Territorium, maximale Tiefe** + fundamentals_drill.md
- [ ] Kapitel 3 Methodology — jede Design-Entscheidung
- [ ] Kapitel 4 Implementation — kompakt
- [ ] Kapitel 5 Results — key_numbers.md
- [ ] Kapitel 6 Discussion + 7 Conclusion
- [ ] Folien-Storyline → PowerPoint (~25–30 Folien + Backup-Folien)
- [ ] Q&A-Drill (Gall-Profil priorisiert)
- [ ] Mock-Defense mit Timing
