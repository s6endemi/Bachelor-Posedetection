# Q&A-Katalog — kumulativ (Verteidigungsfragen mit Musterantworten)

> Wächst mit jeder Sitzung. Alle Antworten auf Englisch drillen (laut!).
> Status: ☐ = noch nicht gedrillt, ☑ = einmal flüssig, ☑☑ = sitzt.

## Drill-10 — die Kurzversionen (einfachstes sprechbares Englisch, zuletzt wiederholen!)

1. **Models:** three criteria — real-time on CPU, mature ecosystems, one per paradigm
   (top-down / bottom-up / one-stage) → behavior connectable to architecture; plus:
   REHAB24-6 authors never tested MoveNet.
2. **NMPJPE:** mean joint error / torso length = % of body size, comparable across
   people & distances. PCK = only hit/miss; OKS = needs COCO tolerances + object area.
3. **Clusters:** frames aren't independent (same video ≈ identical, same person =
   same body) → huge fake sample, tiny meaningless p-values → aggregate to the person:
   median per sequence → average per person → 10 honest points.
4. **Permutation test:** under H0 each person's winner is a coin flip → all 1024 sign
   patterns equally likely → count how many are as extreme as observed = exact
   p-value; all-10-wins → p = 2/1024 ≈ 0.002.
5. **Holm:** 3 tests → false-alarm chance grows; sort p-values, test vs. α/3, α/2, α;
   as strict as Bonferroni, more power.
6. **Frame-independent:** models, not pipelines; smoothing can be added uniformly;
   YOLO has no native temporal processing → any other mode unfair; results = raw
   upper bound.
7. **Numbers:** 10.5 / 12.5 / 12.8 (MoveNet / MediaPipe / YOLO); all pairwise Holm-
   significant; MoveNet wins all 10 clusters.
8. **Coach:** MP–YOLO fits (62 vs 23% exposure, more failures) → MP–MoveNet breaks it
   (16 < 23% exposure, yet 13.8 > 9.1% failures) → MoveNet–YOLO confirms (4× exposure,
   same rates) → mechanism sits at selection stage, left open.
9. **Profiles:** MediaPipe = filtered top-down (coach-safe, rich landmarks / slow,
   weak hips); MoveNet = low-exposure bottom-up (accuracy+stability+speed / least
   complete); YOLO = high-sensitivity one-stage (most complete / most exposed, most
   viewpoint-sensitive).
10. **Healthy volunteers:** target = joint localization, not exercise grading; wrong
    executions included → movement variety covered; NOT covered: patient appearance &
    movement distribution (age, aids, clothing) → can shift pose accuracy → patient
    validation = future work #1.

## A. Übergreifende Klassiker

### ☐ A1: "Why didn't you fine-tune the models on rehabilitation data?"
> "I evaluate off-the-shelf deployment — what a developer can use today. Fine-tuning
> wasn't an option for three reasons: MediaPipe and MoveNet ship as frozen TFLite
> artifacts without official training pipelines, so only YOLO could have been tuned —
> an unfair asymmetry; ten subjects are too few for a clean subject-wise split, and
> tuning on the benchmark would consume it as a test set; and fine-tuned results would
> measure training skill as much as model quality. Fine-tuning on real patient data is
> future work once such data exists."

Fundament: Fine-Tuning braucht (1) gelabelte Trainingsdaten in Menge, (2) sauberen
Train/Test-Split, (3) Zugang zur Trainings-Pipeline. Hier scheitern alle drei:
Asymmetrie (nur YOLO trainierbar), N=10 zu klein + Benchmark würde verbraucht,
Deployment-Frage zielt auf off-the-shelf.

### ☐ A2: "Why exactly these three models?"
> "Three criteria: real-time CPU capability, a mature actively maintained deployment
> ecosystem, and joint coverage of the three architectural paradigms — one
> representative each for top-down, bottom-up, and one-stage. That coverage is what
> lets Chapter 6 attribute behavioral differences to architecture. It also connects to
> prior work: the REHAB24-6 authors evaluated MediaPipe and YOLOv8 but not MoveNet —
> my selection closes exactly that gap while staying comparable."

Ablehnungsgründe für Alternativen (auswendig):
- **OpenPose** — unmaintained, weit von CPU-Echtzeit entfernt
- **PoseNet** — von Google selbst durch MoveNet ersetzt
- **HRNet / ViTPose** — server-grade, nicht mobile
- **AlphaPose** — Multi-Stage-Forschungssystem, kein Mobile-Ökosystem
- **RTMPose** (härteste Nachfrage) — CPU-fähig, aber top-down → Paradigmen-Slot durch
  MediaPipe besetzt; geringere Mobile-Deployment-Verbreitung zum Projektstart;
  Pipeline via Estimator-Interface erweiterbar

### ☐ A3: "Why 2D only? Rehabilitation needs 3D joint angles."
> "2D is the common denominator of this model class — only MediaPipe outputs native
> 3D, and its world landmarks are hip-centered and scale-relative, so a 3D comparison
> would need per-model lifting or alignment machinery that I'd then be evaluating
> instead of the models. The dataset's published ground-truth pathway is the 2D
> projection. For rehabilitation, 3D joint angles are the long-term goal — named as
> future work — but a fair three-family comparison is only possible in 2D."

Fundament: Monokulares 3D ist ill-posed (Tiefen-Ambiguität, nur via gelernte Priors).
MediaPipe world landmarks = metrisch, aber hüft-zentriert/skalenrelativ (GHUM-gestützt);
MoveNet/YOLOv8 rein 2D. Mobile-3D-Marktlage dünn: BlazePose-GHUM *ist* MediaPipe,
Apple Body Tracking ist ARKit-gebunden, Rest Forschungscode.

## B. Statistik (siehe Fundament in ch03_methodology.md §7)

### ☐ B1: "Walk me through your statistical analysis." (30-Sekunden-Antwort)
> "Frames from the same person aren't independent, so I aggregate to the natural
> independence unit — ten subject clusters. On the paired cluster-wise differences I
> run an exact sign-flip permutation test: under the null hypothesis each difference's
> sign is a coin flip, so I enumerate all 2¹⁰ = 1024 sign assignments and count how
> many are as extreme as observed — exact, no distributional assumptions, which
> matters at N=10. Holm correction controls the family-wise error rate across the
> pairwise comparisons with more power than Bonferroni. And percentile bootstrap
> intervals with 20,000 resamples quantify effect size — the test says the difference
> is real, the interval says how large."

### ☐ B2: "Why a permutation test and not a t-test?"
> "A paired t-test assumes the differences are normally distributed — untestable and
> fragile at N=10. The sign-flip permutation test replaces that assumption with a
> symmetry argument: under the null, each sign is exchangeable. With ten clusters I
> can enumerate all 1024 assignments, so the p-value is exact rather than approximate."

### ☐ B3: "Why Holm and not Bonferroni?"
> "Holm controls the family-wise error rate exactly as strictly as Bonferroni but is
> uniformly more powerful — it tests the smallest p-value at alpha over k, the next at
> alpha over k minus one, and so on. There is no scenario where Bonferroni is
> preferable."

### ☐ B4: "What's the smallest p-value you could even observe?"
> "Two-sided, 2 out of 1024 — about 0.002 — when one model wins in all ten clusters.
> That's exactly why I complement the tests with bootstrap intervals: at this cluster
> count the test answers 'is it real', the interval answers 'how large'."

### ☐ B5: "Why not a mixed-effects model?" → siehe ch03 Zone 6.

### ☐ B10: "Why can't you rely on distribution assumptions at n = 10?"
> "A t-test would assume the ten differences are normally distributed. With ten
> values I can neither verify that — normality tests have almost no power at this
> size — nor rely on the central limit theorem. If the assumption is wrong, the
> p-values are wrong. The permutation test replaces the assumption with a symmetry
> argument: under the null hypothesis, the sign of each paired difference is
> exchangeable — and that is guaranteed by the paired design, not assumed."

### ☐ B12: "Why does MultiPose beat SinglePose Thunder (10.5 vs 13.3)? Isn't Thunder the accurate one?"
> "Counterintuitive, yes — Thunder is marketed as the accurate SinglePose variant. My
> explanation — as a hypothesis, the thesis doesn't analyze this mechanism: the
> SinglePose models are designed for a centered, frame-filling person, typically
> achieved through intelligent cropping based on the previous frame. In my
> frame-independent protocol there is no such cropping — the person is small inside
> the 256-by-256 input, so resolution is lost. MultiPose natively processes the full
> image with candidate detection, so it doesn't suffer from that. Consistent with
> this, SinglePose Lightning also lost the most frames to confidence filtering
> (94,395 valid vs. ~119,000)."

Fundament: Backup B5 zeigt die Tabelle + Fußnote. Als EIGENE Hypothese kennzeichnen
(Kap. 6 behandelt SP-vs-MultiPose nicht).

### ☐ B11: "Isn't the 0.1 detection threshold arbitrary? Wouldn't 0.5 have given BETTER accuracy?"
> "With the default of 0.5, MediaPipe's detector refused to report a person in a
> substantial number of frames — in a fixed setup where a person is always clearly
> visible. Those frames would have been booked as detection failures before pose
> quality could even be assessed. Lowering the gate to 0.1 only lets MediaPipe attempt
> more frames — quality is judged afterwards, by the joint-level filter, identical for
> all models. And yes, with 0.5 the measured accuracy would likely have looked
> slightly better — but that's a selection effect: the borderline frames entering at
> 0.1 are the harder ones. The threshold moves errors from the accuracy column into
> the missing-detection column; it doesn't remove them. Since I report completeness
> and failure modes separately, hiding frames behind the detection gate would have
> made the accuracy number less honest — not the model better. Note the direction: if
> anything, this worked AGAINST MediaPipe's accuracy — it didn't manufacture the
> ranking. The exact value is pragmatic rather than tuned; a threshold sensitivity
> study would be the clean quantification — a natural extension."

Fundament: Gate ≠ Judge (Tor lässt antreten, Joint-Filter 0.5 richtet — identisch für
alle). Selektionseffekt: strenges Tor = leichtere Stichprobe = kosmetisch bessere
Accuracy + schlechtere Completeness/Failures → Fehler VERSCHOBEN, nicht entfernt.
Richtungs-Argument: Wahl drückte MPs Accuracy eher (mehr harte Frames rein) → kein
Manipulations-Verdacht möglich, Ranking unabhängig davon. Asymmetrie: nur MP zeigte
Komplett-Ausfälle bei Defaults; Chancengleichheit sitzt an der Bewertungsstufe.

### ☐ A6: "On a phone, MediaPipe easily runs at 30 FPS — are your speed numbers wrong?"
> "The numbers are internally consistent — within every family, latency scales
> monotonically with model size — and they match publicly reported CPU figures. What
> they are not is phone performance: on a phone, MediaPipe runs with GPU delegates
> and in tracking mode, where the detector runs only occasionally. My benchmark
> deliberately measures frame-independent, CPU-only inference through each vendor's
> native stack — a reproducible floor for the 'no accelerator' deployment class,
> measured identically for all nine variants. Absolute numbers shift with
> acceleration; the thesis uses the relative ordering. Repeating this on phones with
> hardware delegates is named as future work."

Fundament: Handy = GPU-Delegate + Tracking-Modus (Detektor läuft selten) + oft
kleinere Auflösung; Thesis = IMAGE mode (Detektor JEDES Frame) + CPU-only + native
Stacks. 100-vs-6-FPS plausibel: kleinstes Ein-Netz-Modell (Lightning, 192², kein
Detektor) vs. größtes Zwei-Netz-System (Heavy = Detektor + größter Estimator).
Konsistenz-Beleg: monotone Latenz-Treppen in jeder Familie (MP 21.9→14.7→6.0;
YOLO 18.9→10.3→4.6). Protokoll: 50 Warm-up, 500 Frames, sequenziell, P95.

### ☐ B6: "You disabled all temporal processing — how can you claim to measure temporal stability?"
> "The models process each frame independently, so what I measure is the temporal
> consistency of the output sequence: consecutive frames are nearly identical inputs,
> and a stable model should return nearly identical outputs plus the true motion. True
> motion is identical across models on the same frame pairs, so between-model
> differences in displacement isolate prediction noise. That's why the thesis
> deliberately calls it a displacement-based stability *proxy* on unsmoothed
> trajectories — it claims exactly what it measures, and because no smoothing is
> applied, it reflects the model's inherent stability rather than a filter's."

Fundament: Stabilität = Eigenschaft der Output-Folge, kein Modell-Mechanismus;
echte Bewegung kürzt sich im Modellvergleich raus (identische Frame-Paare).
Merksatz: Accuracy = Prediction vs. GT („wie weit von der Wahrheit?"); Stability =
Prediction vs. sich selbst einen Frame später („wie sehr zappelt es?"); GT dient bei
Stability nur als Lineal (Torso-Normalisierung), nicht als Vergleichsziel.
Analogie: Drei Wackelkameras filmen dasselbe fahrende Auto — echte Autobewegung in
allen identisch → Unterschiede im Gesamtwackeln = Stativqualität, ohne die echte
Bewegung je zu kennen.

### ☐ B7: "Why is MediaPipe more robust in coach scenes? What's the mechanism?"
(Zwei-Ebenen-Antwort — Thesis-Disziplin + Hintergrundwissen:)
> "The thesis deliberately leaves the mechanism open — the framework can't observe the
> models' internal candidate handling, so Chapter 6 names candidate retention,
> thresholding, and post-detection selection as possibilities without committing to
> one. What the three-way comparison *does* establish is negative: detection count
> alone cannot explain coach robustness — MoveNet exposes fewer secondary persons than
> MediaPipe (16% vs. 23%) yet fails more often. As architectural background beyond the
> thesis: the original BlazePose paper describes its person detector as face-based,
> which would be consistent with suppressing a turned-away therapist — but that's a
> hypothesis I couldn't test, which is exactly why the thesis doesn't claim it."

⚠️ Die Zahlen (16/23/62%, Outlier-Raten 13.8/13.9%) bei Kapitel-5/6-Analyse
verifizieren und in key_numbers.md aufnehmen. → erledigt, siehe key_numbers.md.

### ☐ A4: "Your subjects were healthy volunteers — how is this 'for rehabilitation'?"
(Differenzierte Antwort — trennt die zwei Lücken; Erens eigener Punkt + die echte Grenze:)
> "It's worth separating two things. The evaluation target is joint localization, not
> exercise-correctness classification — and the dataset deliberately includes
> incorrect executions, so movement variability is covered. What healthy volunteers
> cannot cover is the appearance and movement distribution of real patients — age,
> body types, assistive devices, compensatory patterns, everyday clothing instead of
> MoCap suits — and those can shift pose accuracy itself. That's the precise content
> of the limitation, and it's why patient validation is the first item of future work."

Merke: Proxy-Framing im Vortrag = GENAU 2 Sätze (Benchmark-Folie + Limitations) —
Versicherung für den Titel, kein Thema. Nie lockerer claimen als die Arbeit selbst.

### ☐ A5: "You picked YOLO's smallest variant but MediaPipe's middle one — isn't that unfair to YOLO?"
> "The primaries weren't matched by size label — those ladders aren't comparable
> across vendors — but by deployment class: MediaPipe Full at 15 FPS and YOLO Nano at
> 19 FPS sit in the same CPU real-time band, and both are the vendor-default entry
> points. The variant analysis keeps this transparent: yes, YOLO Small beats MediaPipe
> Full on accuracy. But it wouldn't change the headline — MoveNet MultiPose dominates
> every YOLO and MediaPipe variant on accuracy AND speed simultaneously, so the
> ranking and the three-profiles conclusion are robust to that choice."

Fundament: Small 11,43% @ 10,3 FPS · Medium 10,86% @ 4,6 FPS · MP_Heavy 11,51% @ 6,0
— alle von MultiPose (10,52% @ 27,7) auf BEIDEN Achsen dominiert; Pareto-Front =
{MultiPose, SP_Lightning}. Ehrliche Grenze: Coach-/Completeness-/Failure-Analysen
liefen nur mit den Primaries (Rechenzeit-Scope) — für Small nicht gemessen.

### ☐ B9: Follow-ups auf den selbst benannten Paradigma-Confound (Folie 17)
Prinzip: Der Confound steht in der Thesis (6.2) — ihn im Vortrag zu benennen
VERBRAUCHT den Angriffsvektor (wer zuerst benennt, kontrolliert das Framing).
Angriffsfläche entsteht durch Über-Claims und verschwiegene Schwächen, nie durch
präzise benannte Grenzen.

**"If you can't separate architecture from training data, why present it at all?"**
> "Because it's a structured, falsifiable hypothesis. The three signatures are exactly
> what each paradigm's design logic would predict — commit-then-predict fails
> confidently, per-joint confidence degrades gracefully, coupled detection fails
> completely. That consistency is worth reporting, and it tells future work precisely
> what to test."

**"How would you disentangle it?"**
> "Two ways: evaluate several model families per paradigm — ideally retrained on
> identical data — and log the models' internal candidate states instead of observing
> only external behavior. Both are named as future work."

### ☐ B8: "Why is MediaPipe's hip error so high? Training data?"
(Erens eigene Herleitung 7.7. — deckt sich mit Kap. 6.4 + Trainingsdaten-Confound aus 6.2)
> "The error is measured against the MoCap joint center. MoveNet and YOLOv8 are trained
> on COCO-style hip annotations, which appear closer to that convention, while
> BlazePose is trained on Google's own dataset with its own 33-landmark scheme derived
> from a body model. So MediaPipe's hip error is plausibly in part a definition
> mismatch rather than a perception failure — the thesis supports this indirectly: the
> error is concentrated in one structurally specific region, and excluding the hips
> closes about 40% of the gap. Against a ground truth following BlazePose's convention,
> the hip error would likely shrink. For rehabilitation it still matters, because
> anatomical joint angles are the target — though a systematic offset could in
> principle be calibrated away, which I'd note as an untested extension."

Fundament: 4 Hüft-Konventionen im Spiel — COCO-Annotation (MoveNet/YOLO-Training),
BlazePose-Schema (körpermodell-basiert, Google-intern), MoCap-Gelenkzentrum (GT),
jeweils ≠. Trainingsdaten NICHT identisch: MoveNet = COCO+Active, YOLO = COCO,
BlazePose = eigenes Set. Bonus: derselbe Mechanismus erklärt plausibel einen Teil von
MediaPipes COCO-Absturz (fremde Annotationskonvention = Auswärtsspiel) — als Hypothese.
Nuancen: (1) trotzdem deployment-relevant (anatomische Winkel!), (2) systematisch →
prinzipiell kalibrierbar (eigene ungetestete Erweiterung, NICHT Thesis-Claim).
Nicht proaktiv auf Hauptfolie — Backup B8 + diese Antwort drillen.
