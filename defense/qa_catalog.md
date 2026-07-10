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
   the YOLO *network* has no temporal processing — tracking is a pipeline layer on
   top (not used); results = raw upper bound.
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

### ☐ B12: "How did you measure X?" — die Mess-Rezepte (Struktur: was + Daten + Rechnung)
- **Accuracy:** Euclidean distance predicted↔MoCap joint, averaged over 12 joints, /
  torso length = NMPJPE; valid subset; median per sequence → mean per person → tests.
- **Per-Joint:** same NMPJPE per region; sensitivity: recomputed without the two hips.
- **Viewpoint:** orientation label from 3D MoCap shoulders, converted per camera,
  10°-bins; compare frontal (0–20°) vs. lateral (60–90°).
- **Stability:** distance prediction(t)↔prediction(t+1) / avg torso of both frames;
  joint counts only if valid in both; averaged over joints & pairs.
- **Multi-Person:** (a) metadata subset severity≥2: accuracy inside vs. outside;
  (b) coach subset (5 videos, manual review): share of frames with 2+ persons
  reported (every 10th frame, ~1,200) + catastrophic rate (>100% torso).
- **Completeness:** after confidence filter, count valid joints per frame; share of
  12/12 frames + average count.
- **Speed:** dedicated benchmark — 3 videos × 500 frames + 50 warm-up, sequential,
  CPU-only, high-res timers, native stacks; mean/median/P95 latency.
- **Failure-Modes:** each failure frame → exactly one of 4 categories, fixed order:
  no detection / <6 valid joints / outlier + visible 2nd person / outlier without.
- **COCO:** 1,519 images, nearest-torso-center matching, visible GT joints only,
  same 12 joints, descriptive.
Bei Implementation-Nachfrage: "two-phase pipeline — predictions stored per frame,
metrics computed deterministically in phase 2; every number traceable."

### ☐ B11: "But are these values actually acceptable — good enough in absolute terms?"
Muster: (1) übersetzen (10.5% Torso ≈ ~5 cm Gelenkfehler, "roughly, for intuition"),
(2) "acceptable for WHAT?" — Task definiert Schwelle, (3) Thesis claimt bewusst keine
klinische Schwelle (bräuchte definierte Anforderung + Patienten) → liefert
Größenordnungen + Vergleichsbasis.
> "The values are relative to torso length — roughly, 10.5 percent is about five
> centimeters of average joint error on an adult. Whether that's acceptable depends
> on the task: for repetition counting and coarse form feedback, comfortably yes; for
> precise clinical joint angles, borderline — centimeters translate into degrees. My
> thesis deliberately doesn't claim a clinical threshold — that needs a defined
> requirement and patient validation, which is the future work. It provides the
> orders of magnitude and the comparison basis. And one number is clearly NOT
> acceptable without countermeasures: 9–14% catastrophic coach failures — which is
> why deployment guidance matters as much as model choice."

Pro Dimension: NMPJPE ~5cm → reps/grobes Feedback ja, klinische Winkel grenzwertig ·
Displacement: kein absoluter Schwellwert (Proxy, enthält echte Bewegung) — Vergleich
zählt; Glättung senkt weiter · Rotation +18%: kein Zusammenbruch → Deployment-Regel
(frontal anleiten) · Completeness 38%: 62% der Frames unvollständig → Lückenbehandlung
nötig, deshalb eigene Dimension · Coach 9–14%: klar inakzeptabel ohne Monitoring
(offensiv sagen = Stärke) · 27.7 FPS: klar ausreichend (<~10 FPS = träge → Heavy/
Medium fallen durch).

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

### ☐ B9: Follow-ups auf den selbst benannten Paradigma-Confound (Folie 20)
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

## C. Confound- & Ground-Truth-Fragen (Gall/Krüger-Kaliber — Session 10.7.)

> Diese Fragen greifen die METHODE an, nicht die Zahlen. Muster immer gleich:
> (1) den Punkt ZUGEBEN, (2) zeigen, warum der VERGLEICH trotzdem hält,
> (3) den sauberen Test als Extension benennen. Nie leugnen, nie bluffen.

### ☐ C1: "You evaluate each model only on its own confident joints. MoveNet keeps the fewest. Isn't its accuracy win just an artifact of dropping the hard joints?" (GEFÄHRLICHSTE FRAGE)
**Heißt einfach:** „MoveNet lässt die schweren Gelenke weg und wird nur auf den leichten gemessen — ist sein Genauigkeits-Sieg also ein Filter-Trick?"
> "Yes, that link is real — accuracy is only measured on the joints that pass the
> filter, and MoveNet keeps the fewest. But three things. First: the filter is the
> same 0.5 for every model, and it matches real use — an app also works only with
> the confident joints. Second: it does not look like MoveNet simply drops the hard
> joints. It leads in five of six body regions and on all six exercises — not only
> where a lot gets filtered. Third: the clean check would be to score all models on
> the same joints — only those that are valid for all three at once. That is not in
> the thesis; it is a natural next step. I expect the order to stay and the gap to
> get a bit smaller. And that is exactly why completeness is its own dimension — I
> show the trade-off instead of hiding it."

Fundament: Bias-Richtung sofort zugeben (nimmt dem Angriff die Spitze). Gegenindizien:
5/6 Regionen, 6/6 Übungen, alle 10 Cluster. Intersection-Analyse = der saubere Test
(nicht in der Thesis!). Completeness als eigene Dimension = die ehrliche Buchführung —
RQ1 trennt die beiden bewusst ("separates accuracy from completeness", Kap. 6.1).

### ☐ C2: "MoveNet filters its shakiest joints away — so your stability number rewards abstaining. And at 10 Hz there is real motion inside your displacement."
**Heißt einfach:** „Belohnt deine Stabilitätszahl das Weglassen wackliger Joints? Und steckt bei 10 Hz nicht echte Bewegung im Wert?"
> "Both points are fair — that is why I call it a stability proxy, not jitter. A
> joint only counts if it is valid in two frames in a row. So yes, the metric only
> sees the surviving joints, and I cannot fully rule that effect out. But one thing
> speaks against it: all three MoveNet variants take the top three ranks — and they
> have very different completeness levels. On the 10 Hertz point: every model sees
> exactly the same frame pairs, so the real motion inside the number is the same for
> all of them. The comparison stays fair — only the absolute values are not 30-fps
> jitter."

Fundament: Verbindung zu Folie 13 — dort bewusst "mainly reflect prediction noise"
sprechen. Das Confidence-Collapse-Profil hilft sogar: MoveNet enthält sich statt zu
raten — das verbessert Stability UND verschlechtert Completeness. Genau das Profil.

### ☐ C3: "MoveNet looks like CenterNet — center heatmap plus regression. Classic bottom-up means keypoint detection plus grouping. Is your label even correct?"
**Heißt einfach:** „Ist ‚bottom-up' für MoveNet überhaupt das richtige Label?"
> "Google itself calls MoveNet bottom-up, and I follow the taxonomy of the Zheng
> survey. You are right: it has no classic grouping step like part affinity fields —
> the person-center decoding replaces that. For my analysis, the label matters less
> than the properties: one pass over the full image, no cropping, no detector that
> commits first, and a confidence per joint. Those properties are what my failure
> analysis connects to. If you prefer to call it center-based one-stage — fine. None
> of my conclusions change."

Fundament: Nicht am Label festbeißen — auf die Eigenschaften umlenken. Backup B9 hat
die CenterNet-Details (4 Heads, inverse-distance-Decoding).

### ☐ C4 (Krüger): "Your denominator is the projected 2D torso length. When the person bends forward, the torso shrinks in the image and your error percentage gets inflated."
**Heißt einfach:** „Bei einer Vorbeuge schrumpft die Torsolänge im Bild — dein Nenner — und bläht die Fehlerprozente auf."
> "That is true — the normalization depends on pose and view. Three things keep it
> under control. First: within one frame, the denominator is the same for all three
> models — so the comparison and the ranking are not affected. Second: the absolute
> statements — 10.5 percent is roughly five centimeters — are rough orders of
> magnitude, not clinical thresholds. Third: I use the median per sequence, and that
> damps extreme frames. The alternatives are not better: a scale from the prediction
> would be circular, and a bounding box also changes with posture. A sensitivity
> check over different normalizations would be a good extension."

Fundament: Seitliche Drehung (Yaw) verkürzt den Torso kaum (vertikale Strecke) —
kritisch sind Rumpfbeugen. Falls als Teilerklärung für Ex3/Ex5-Fehlerhöhen gefragt:
nur als plausible Hypothese, nicht als Fakt.

### ☐ C5: "On COCO you match predictions to the ground truth — on REHAB you don't. Why no ablation with ground-truth matching on REHAB, to separate selection errors from pose errors?"
**Heißt einfach:** „Warum kein Kontroll-Lauf, in dem die Pipeline immer die RICHTIGE Person nimmt? Dann wüsstest du, wie viel Coach-Fehler nur an der Personenauswahl liegt."
> "That was a deliberate split. On REHAB I simulate the real app — and a real app
> has no ground truth, so it must pick the person on its own. On COCO there is no
> patient, so I have to match to the annotated person — there it is unavoidable.
> But yes: I could run REHAB a second time, where the pipeline always picks the
> person closest to the ground truth — a perfect selection. The difference between
> that run and my normal run would show exactly how much of the coach failure comes
> from picking the wrong person, and how much from bad poses. That is a clean
> extension, and it only needs the evaluation phase — no new inference. My guess:
> MoveNet and YOLO would improve a lot, MediaPipe much less."

Fundament (VERSTÄNDNIS, ganz einfach): „Oracle selection" = ein Kontroll-Lauf, in dem
die Pipeline schummeln darf. Wenn mehrere Personen erkannt werden, nimmt sie immer
die, die der Ground Truth am nächsten liegt — also garantiert die Patientin, nie den
Coach. Damit verschwinden ALLE Auswahlfehler; was an Fehlern übrig bleibt, sind echte
Posefehler. Vergleich „normaler Lauf vs. Schummel-Lauf" = wie viel Prozent der
Coach-Fehler NUR an der Personen-AUSWAHL hängen. Warum nicht gemacht: REHAB soll
bewusst die echte App simulieren — und eine App hat keine Ground Truth. Technisch
wäre es billig (nur Phase 2: gespeicherte Predictions neu auswerten, keine neue
Inferenz) → deshalb selbstbewusst als saubere Extension anbieten. Erwartung begründen
mit Panel (c) der Coach-Figur: MoveNet erkennt dort BEIDE Personen korrekt und wählt
trotzdem den Coach → das ist ein reiner Auswahlfehler, den der Oracle-Lauf beheben
würde.

### ☐ C6 (Krüger): "How accurate is your ground truth itself? Markers are millimeter-accurate, but marker-to-joint-center offsets are centimeters. Your winner is at about five centimeters — that is uncomfortably close."
**Heißt einfach:** „Wie genau ist deine Ground Truth selbst? Der Versatz zwischen Marker und echtem Gelenkzentrum ist Zentimeter groß — und dein bestes Modell liegt selbst nur bei ~5 cm."
> "The marker positions themselves are millimeter-accurate. The real uncertainty
> comes from two steps: where the skeleton model places the joint center relative to
> the markers, and the projection into the image. These effects are mostly
> systematic — they shift all models against the same reference. That is why I treat
> the error levels as a fair comparison basis, not as absolute truth. For the
> ranking, a shared offset cancels out. For the absolute numbers you are right: five
> centimeters of model error against a reference with maybe one or two centimeters
> of its own uncertainty — that means real error bars. That is one reason the thesis
> makes no absolute clinical claims."

Fundament: Systematisch vs. zufällig trennen — systematische GT-Offsets treffen alle
Modelle gleich (Ranking robust), machen aber Absolut-Level unscharf (deshalb keine
klinische Schwelle). Ehrliche Nuance falls nachgehakt: MoveNet/YOLO sind auf
COCO-Konventionen trainiert, die der MoCap-Joint-Center-Konvention näher liegen →
die GT-Konvention ist nicht 100 % neutral — führt direkt zur Hip-Antwort B8.

### ☐ C7 (Krüger): "How was the MoCap projected into the camera, and how good is the synchronization?"
**Heißt einfach:** „Wie kam das MoCap-Skelett ins Kamerabild, und wie gut passen Video und MoCap zeitlich zusammen?"
> "I use the 2D ground truth exactly as the dataset publishes it: the authors build
> the 26-joint skeleton from the 41 markers and project it into each camera view —
> as far as the paper describes it, with estimated extrinsics, not a full lab
> calibration. I checked the result visually by overlaying the skeletons on the
> frames — they sit on the person. The synchronization also comes with the dataset.
> Whatever small projection or sync error remains, it is the same reference for all
> three models — so the comparison holds. Sync errors matter most for fast motion,
> and these exercises are slow."

Fundament: Nie mehr Detail claimen, als das Dataset-Paper hergibt ("as far as the
paper describes it"). Rückfalllinie = Notfall-Satz: "I'd have to check the paper for
the specifics — what I took from it was …".

### ☐ C8: "In your multi-person subset, MoveNet and YOLO get BETTER with a second person in the frame — minus five percent. How can more people make a model better?"
**Heißt einfach:** „Wieso wird MoveNet mit zweiter Person im Bild angeblich BESSER (−5 %)? Das ergibt doch keinen Sinn."
> "It doesn't — and the thesis makes no causal claim there. The multi-person frames
> are not a random sample. They come from specific videos, exercises and moments. So
> the small differences — plus one, minus five, minus four percent — mostly reflect
> WHICH frames are in that subset, not an effect of the second person. That is
> composition, not causation. The honest message of that slide is: a second person
> somewhere in the frame does not hurt by itself. The real risk is the coach
> scenario — and that one I analyze separately."

Fundament: Folie 16 sagt selbst "(composition effects — not causal)" — die Antwort
zitiert nur die eigene Folie. Das Minuszeichen NIE als "Modell wird besser" verkaufen.

### ☐ C9: "Your three models differ in input resolution and training data. How much of your architecture story is really resolution or data?"
**Heißt einfach:** „Wie viel deiner Architektur-Story ist in Wahrheit nur Auflösung oder Trainingsdaten?"
> "I cannot separate those — and the thesis says so openly. Each paradigm is
> represented by one family, so architecture, training data and input size always
> move together. That is exactly why the failure-signature result is labeled as an
> observation and a hypothesis — not a mechanism. What I can say: I test the models
> as they ship, because that is the decision a developer actually faces. To untangle
> it, you would need several families per paradigm, ideally retrained on the same
> data — and that is named as future work."

Fundament: Erweiterung von B9 um die Input-Resolution-Facette (192²/256² vs. 640).
"As they ship" ist die Forschungsfrage selbst, kein Notausgang.

### ☐ C10: "Why a sign-flip permutation test and not Wilcoxon?"
**Heißt einfach:** „Warum kein Wilcoxon-Test?"
> "Wilcoxon would also work — it is exact at this sample size too; it is basically
> the same idea, just on ranks. I flip the raw differences instead, because that
> keeps the real effect sizes inside the test and fits naturally with the bootstrap
> intervals. Both give the same answer here."

### ☐ C11: "Why the median per sequence, but the mean across sequences and clusters?"
**Heißt einfach:** „Warum innerhalb eines Videos der Median, aber darüber der Mittelwert?"
> "Two different jobs. Inside a sequence I take the median, so a few catastrophic
> frames cannot dominate a video's score — that is outlier protection. Across
> sequences and people I take the mean, because at that level every video and every
> person should count proportionally. So: median against outlier frames, mean for
> the aggregation."

Fundament: Genau diese Wahl erklärt auch, warum der Cluster-Gap MP−MoveNet (1,01pp)
kleiner ist als der Frame-Gap (2,01pp): Sequenz-Mediane dämpfen MediaPipes schwere
Fehler-Frames (Std 7,05 vs. 4,66) → die Folie-10-Falle ("Ihre Zahlen passen nicht
zum Boxplot") genau damit beantworten.

### ☐ C13: "You lowered MediaPipe's detection threshold to help it — but left MoveNet's low completeness untouched, 'out of the box'. Isn't that inconsistent?"
**Heißt einfach:** „Bei MediaPipe hast du am Regler gedreht, bei MoveNet nicht — misst du da mit zweierlei Maß?"
> "No — those are two different kinds of thresholds, and my rule was the same for
> all models. MediaPipe's 0.1 is the entry gate: with the default, the model
> reported no person at all in many frames — in a fixed setup where a person is
> always clearly visible. I cannot judge pose quality on a frame the model refuses
> to attempt. That is an obstacle to measurement, so I opened the gate. MoveNet's
> low completeness is different: that is the model's own confidence behavior on
> single joints — a result, not an obstacle. The quality judgment itself — the 0.5
> joint filter — is identical for all three models, and I never touched it. So the
> rule is: remove obstacles to measurement, never adjust the measurement itself.
> And note the direction: opening MediaPipe's gate let harder frames in — if
> anything, it hurt MediaPipe's accuracy number, it did not create the ranking."

Fundament: Tor vs. Richter (B11-Logik, jetzt symmetrisch zu Ende gedacht). MediaPipe-
Eingriff = Messhindernis entfernen (Komplett-Ausfälle trotz sichtbarer Person =
Artefakt). MoveNet-Completeness = Messergebnis (Konfidenz-Kalibrierung IST das
Verhalten). Der gemeinsame Richter (Joint-Filter 0.5) blieb unangetastet →
konsistente Regel, kein zweierlei Maß. Würde man MoveNet „helfen", müsste man den
RICHTER senken — das änderte die Bewertung für alle.

### ☐ C14: "Why keep accuracy and completeness separate? Couldn't you combine them into one score — and how?"
**Heißt einfach:** „Warum zwei getrennte Zahlen? Und wenn kombinieren — wie genau?"
> "I keep them separate on purpose. Any combined score must pick a weighting
> between the two — and that weighting depends on the application. A rep counter
> barely cares about missing joints; a joint-angle tracker cares a lot. My goal was
> the decision basis, not the decision. Also important: thresholds do not remove
> errors, they only move them between columns — accuracy, completeness, failure
> modes. That is why the thesis reports every column. If you do want one combined
> view, there are two clean ways — both good extensions. First: sweep the joint
> filter from low to high and plot accuracy against completeness — then each model
> becomes a trade-off curve, and you compare curves instead of one operating point
> at 0.5, like precision–recall curves. Second: a miss-penalized metric — score all
> twelve ground-truth joints in every frame, and a joint the model does not report
> counts as a miss, PCK-style. Then abstaining costs points, and one number covers
> both. And honestly: how the ranking would look under that metric is genuinely
> open — MoveNet keeps 86 percent of the joints and is accurate on them, YOLO keeps
> 96 percent and is less accurate. That is exactly why it would be a meaningful
> extension, not a formality."

Fundament: Kern-Prinzip = kombinierte Metrik erzwingt EINE Anwendungs-Gewichtung →
Thesis liefert bewusst die gewichtungsfreie Entscheidungsbasis (Drei-Profile-Logik).
„Fehler verschwinden nicht, sie wechseln die Spalte" = derselbe Satz wie in B11.
Methode 1 (Sweep→Kurve) ist konsistent mit Bestehendem: B11 nennt die threshold
sensitivity study schon als Extension; Failure-Taxonomie-Robustheit (Cutoff 4–8) ist
eine Mini-Version. Methode 2 (Miss-Penalty/PCK über alle GT-Joints) = wie COCO
fehlende Keypoints behandelt. Zahlen: valid joints/frame 10,31 (MoveNet, 86 %) ·
10,72 (MP, 89 %) · 11,53 (YOLO, 96 %). Ausgang offen sagen = Stärke.

### ☐ C12: "You mention a video-level sensitivity check — where is that in the thesis?"
**Heißt einfach:** „Wo genau steht das Video-Level-Ergebnis in der Arbeit?"
> "The methodology names it as a sensitivity check; the result itself is in the
> analysis artifacts of my repository — not as a table in Chapter 5. What it shows:
> the ranking and the significance pattern stay the same at video level. The chapter
> reports the cluster level, because that is the honest unit for inference."

Fundament: NICHT auf eine Kapitel-5-Tabelle verweisen — es gibt keine. Die
Frame-Level-Übereinstimmung dagegen steht in der Thesis ("the same ordering remained
after subject-cluster aggregation", Kap. 5.2).
