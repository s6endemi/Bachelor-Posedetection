# Mündliches Skript — Defense (FINAL v5, 26 Folien — DECK EINGEFROREN)

> Tempo: ~120 Wörter/Min; 2 Sek. Pause nach jedem Titel; Signposts (Ende Folie 9 und
> 14) absetzen; auf Figuren ZEIGEN. Nominal ~34 Min → bei Deinem Tempo real ~27–28.
> Live zu lang? Skippen: Folie 19 (ein Klick, nichts bricht). Rubber: 12, 21, 23.
> NIE kürzen: Folie 18 (Dreisatz) und 22 (Profile).

---

## Folie 1 — Titel (0:30)

Good morning, and thank you for being here. My name is Eren Demir. Today I am
defending my bachelor thesis: a holistic evaluation of 2D mobile pose estimation
models for home-based rehabilitation.

Physiotherapy works when exercises are done correctly and regularly — but most of
it happens at home, unsupervised, and adherence is poor. A smartphone camera with a
pose estimation model could close that gap. So my thesis asks: which of today's
mobile models can you actually trust in a living room? And the answer turned out to
be more interesting than a simple ranking.

---

## Folie 2 — The problem (1:15)

Why does this matter? Studies show that many patients skip their home exercises. And
when they skip them, recovery takes longer and costs go up.

Pose estimation offers a practical solution. A single RGB camera — the one in every
smartphone — can estimate where the body joints are. No markers, no suits, no lab
needed. [aufs Diagramm zeigen] The patient exercises at home. The camera and the
model watch. The system gives feedback and tracks progress.

The open question is the model itself: which mobile pose estimator can actually
carry such a system? That is the question of this thesis.

---

## Folie 3 — Four deployment constraints (2:00)

There are already benchmarks for pose estimation. COCO and MPII rank models every
year. However, those benchmarks answer a different question. Home rehabilitation has
four conditions they do not cover.

[Boxen nacheinander] One: camera position. Patients rarely position themselves
frontally — you get oblique and side views, and self-occlusion. Two: other
people. A family member or a therapist walks into the frame — and the system must
keep tracking the right person. Three: stability over time. Feedback runs on
video. If the skeleton jitters from frame to frame, the feedback feels broken.
Four: hardware. There is no GPU in a living room. The model must run on a normal
CPU.

These four constraints define the evaluation — each one becomes a measured
dimension. Together with accuracy and completeness, that gives six dimensions, and I
will go through them one by one.

---

## Folie 4 — Research questions (0:30)

Formally, this gives five research questions. You can read them simply as: how
good, how stable, how robust, does it transfer — and what should we deploy. I will
answer all five by the end.

---

## Folie 5 — The scope of this evaluation (0:50) [skippbar]

Before the setup, one slide on the scope. Nine model configurations — three
families, three variants each — all under one shared protocol. That meant over
1.1 million inference runs. Six quality dimensions, plus a
cross-dataset check on COCO. And to make the models comparable at all, three
different landmark systems — 33, 17, and 26 joints — had to be unified into one
shared 12-joint skeleton.

In short: not a new model — a decision basis that did not exist before.

---

## Folie 6 — Three candidates, three construction principles (3:00)

These are the three candidates. Each one represents one of the three main ways to
build a pose estimator.

[links] MediaPipe is top-down. Two stages: first a detector finds the person.
Then the image is cropped, and a second network estimates the pose on that crop.
Because it sees the person in full resolution, it is very precise per person. The
price: the cost grows with every person in the image. And everything depends on the
detector — no detection means no pose. It also gives 33 landmarks, more than the
others.

[Mitte] MoveNet is bottom-up. One single pass over the whole image. It predicts
person centers as a heatmap, and gets the keypoints from those centers. No cropping,
no second network. The cost stays the same no matter how many people are in the
image. The weak spot: grouping — deciding which keypoints belong to which person.
Internally, MoveNet refines each keypoint with full-image heatmaps, weighted towards
the person center — a detail that becomes relevant in the multi-person analysis
later.

[rechts] YOLOv8-Pose is one-stage. It is an object detector with an extra pose
head. Every detection comes out with its box and all 17 keypoints — in one shot. Fast
and simple. But it has no high-resolution zoom on the person, like top-down has.

I selected by three criteria: real-time on CPU, a mature ecosystem, and this paradigm
coverage — because it lets me connect behavior to architecture later. Six family
variants join the comparison as well.

---

## Folie 7 — REHAB24-6 (2:30)

The testbed is REHAB24-6 — a public benchmark of real physiotherapy exercises. Ten
healthy adults. Six exercises, guided by a physiotherapist. There are two
synchronized cameras — one roughly frontal, one roughly lateral.

The key feature is the ground truth. [aufs Bild] The subjects wore motion-capture
suits with 41 markers. From those, the system derives a skeleton and projects it into
the camera image. So for every frame, I know where every joint really was. The
green dots are the 12 joints all models share — shoulders, elbows, wrists, hips,
knees, ankles. Those 12 are what I evaluate.

Of course, this ground truth has its own error sources — marker placement, the
skeleton model, the projection into the image. That is exactly why I use it as a fair
comparison basis: all models are measured against the same reference.

[auf die Person im Hintergrund zeigen] And note the second person in the background —
this becomes relevant later.

One important qualification: this is a proxy benchmark with healthy volunteers,
not a clinical cohort. So it is a rehabilitation-oriented testbed — not clinical
validation. I will come back to that at the end.

---

## Folie 8 — The protocol (2:15)

To compare fairly, every model runs through one fixed protocol. [Pipeline entlang]
We take every third frame — 10 frames per second. Every model sees every frame
independently. No tracking, no smoothing, no memory. If a model detects more than
one person, the pipeline selects the largest one — the most prominent person, like a
real app would. All predictions are mapped to the same 12 joints. Joints with
confidence below 0.5 are dropped — so unreliable guesses do not distort the accuracy
numbers. Then we compute the metrics.

The main metric is NMPJPE — normalized mean per joint position error. In plain
terms: the average distance between predicted and true joint positions, divided by
the person's torso length. So it is an error in percent of body size — comparable
across people, distances, and resolutions.

The evaluation is frame-independent because I evaluate models, not pipelines.
Smoothing and tracking — such as a Kalman filter — can be added on top of any model,
but they would mix engineering into the model comparison.

---

## Folie 9 — Honest statistics (1:15)

One slide on the statistics — each choice here solves a specific problem. [Zeilen
entlang] We have 367,000 frame observations — but frames of the same person are not
independent. So the analysis aggregates to ten subject clusters: the median error
per sequence, averaged per person.

On these cluster differences I run paired permutation tests. For me, that is the
natural choice: with only ten data points, I cannot rely on distribution assumptions
— and a permutation test needs none. Even better: at this sample size, the test is
exact — all 1024 combinations can simply be checked.

Because I compare three model pairs, Holm correction keeps the false-alarm rate
under control. And bootstrap confidence intervals add the second half of the
answer: not just whether a difference is real, but how large it is. As a sensitivity
check, I repeated the analysis on the video level — the ranking stays the same.

That completes the setup: six dimensions, nine model configurations, one fixed
protocol. Now to the results.

---

## Folie 10 — Dimension 1: Accuracy (2:00)

Dimension one: accuracy. [Boxplot] Every dot here is one person — their average error
over all their videos. The overall numbers: MoveNet 10.5 percent. MediaPipe: 12.5.
YOLOv8: 12.8. To make that concrete: ten percent of torso length is roughly five
centimeters on an adult.

And this is consistent across subjects: MoveNet's whole distribution sits below the
other two. In fact, MoveNet wins in every single cluster, against both
competitors — the strongest outcome the exact test can produce. All differences
survive Holm correction, and MoveNet is first on all six exercises.

So on accuracy: a clear winner.

---

## Folie 11 — Accuracy by body region (1:15)

One level deeper: where do these errors actually sit? [Tabelle] Not uniformly.
MediaPipe is actually the best model on the shoulders. MoveNet leads everywhere
else. But look at the hips: MediaPipe is at seventeen percent — against eleven for
MoveNet. That is a concentrated problem, not general weakness.

And there is a plausible reason: every model family defines the hip slightly
differently. MoveNet and YOLO are trained on COCO annotations. MediaPipe on Google's
own landmark scheme. And my ground truth is the motion-capture joint center — a third
convention. If I remove the hips, MediaPipe's gap to MoveNet shrinks by about forty
percent. So part of this is likely a definition mismatch, not a perception
failure — the evidence is indirect, and the thesis marks it that way.

---

## Folie 12 — Dimension 2: Viewpoint (0:45)

Dimension two: what happens when the patient turns sideways? All models get worse —
side views mean self-occlusion. But not equally: MoveNet degrades by 18 percent,
MediaPipe by 28, YOLOv8 by 32. One note on the method: the body angles come from the
3D motion capture, calibrated to each camera — not from the predictions, so model
errors cannot distort this analysis. Again: MoveNet.

---

## Folie 13 — Dimension 3: Stability (1:00)

Dimension three: temporal stability. Here I measure how much the predictions move
between two consecutive frames — the normalized frame-to-frame displacement, in
percent of torso length. This runs on raw outputs, without any smoothing. And since
the true motion is the same for every model on the same frames, the differences
between the models mainly reflect prediction noise.

The result: MoveNet 3.8 percent. YOLOv8 5.2. MediaPipe 6.0. Across all nine variants,
the whole MoveNet family takes the top three — which suggests this stability comes
from the architecture itself, not from any filter.

---

## Folie 14 — Dimension 4: Speed (1:15)

Dimension four: CPU speed. [Scatter] This is the accuracy-speed landscape of all nine
variants. Down and right is better. Among the main models, MoveNet leads with about
28 frames per second. YOLOv8: 19. MediaPipe: 15. If you only need speed, MoveNet
SinglePose Lightning reaches 100 FPS. The bigger variants pay a real price: MediaPipe
Heavy and YOLOv8 Medium reach better accuracy, but at 6 and 5 FPS — below real time.

The scatter also shows the families themselves. Within each family, more parameters
buy accuracy and cost speed. And one surprise: MoveNet MultiPose beats both
SinglePose variants. One plausible reason: SinglePose models expect one centered
person — and in frame-independent mode, the subject is small in their input. That
makes MultiPose the only configuration on the efficiency frontier, together with
Lightning.

[Pause. Langsam:] At this point, MoveNet leads in all four dimensions: accuracy,
viewpoint, stability, speed. The remaining two dimensions change that picture.

---

## Folie 15 — Dimension 5: Completeness (0:45)

Dimension five: how often does a model deliver the complete 12-joint skeleton?
Here, the order flips. YOLOv8: 79 percent of frames. MediaPipe: 55. MoveNet — the
accuracy leader — only 38. If your application needs every joint in every frame, the
winner just changed.

---

## Folie 16 — Dimension 6: Multi-person, two regimes (1:00)

Dimension six: other people in the frame. This question has two very different
answers, so I analyzed it in two parts. [links] First, the broad view: in all frames
where a second person is somewhere visible, accuracy barely changes — just having
another person in the frame is not a problem. [rechts] But five recordings are
different: there, the therapist stands prominently in the picture — the situation I
showed on the benchmark slide. In this coach scenario we see catastrophic
failures: the model switches to the wrong person. These are two different regimes —
and if you average over both, you hide the real deployment risk.

---

## Folie 17 — One frame, three behaviors (1:30)

Let me show you one coach frame — the same moment, three models. [Panel b] MediaPipe
finds only the patient. The therapist does not even appear. [Panel c] MoveNet detects
both people — detection works — but its selection rule puts the box on the
therapist. That is a selection failure, not a detection failure — this
distinction matters. [Panel d] YOLOv8 detects three candidates — and still picks the
patient correctly.

One frame, three completely different behaviors. MediaPipe reports a second person in
23 percent of coach frames. MoveNet: 16. YOLOv8: 62. And the catastrophic failure
rates: 9, 14, and 14 percent.

---

## Folie 18 — The obvious explanation fails (2:00) [LANGSAM — Höhepunkt]

These numbers point to one central finding of this thesis. The intuitive explanation
would be: models that detect the second person more often also fail more often.
The three pairwise comparisons put this to the test.

[Zeile 1] MediaPipe versus YOLOv8: MediaPipe reports the second person in 23 percent
of frames, YOLOv8 in 62 — more detections, more failures. This pair is consistent
with the explanation.

[Zeile 2] But MediaPipe versus MoveNet is not. MoveNet reports the second person in
only 16 percent of frames — less than MediaPipe — so it should be safer. Instead, it
fails more: 13.8 versus 9.1 percent. Here the explanation fails.

[Zeile 3] And MoveNet versus YOLOv8 confirms this: four times more detections, yet
nearly identical failure rates.

The conclusion: detection count alone does not explain coach robustness. The
mechanism must sit deeper — at the selection stage, in how candidates are kept and
chosen. Since the models do not expose those internals, the thesis names the
candidates and leaves the mechanism open.

---

## Folie 19 — One video breaks all three (0:50) [skippbar]

One more honest look at those five coach videos — they are not all equal. [Tabelle]
In four of them, MediaPipe stays below 40 percent error, while the others fail. But
one video — PM 010 — breaks all three models, including MediaPipe: over 65
percent error everywhere.

So: no selection strategy is perfect. MediaPipe is the most robust of the three — but
not immune.

---

## Folie 20 — Failure signatures (1:30)

The failure analysis completes the picture. I classified every failure frame into
four categories — and each model is dominated by a different one. MediaPipe's
dominant category is Keypoint-Displacement — confident misprediction: wrong
keypoints, high confidence. MoveNet's is Confidence-Collapse — graceful
degradation: it filters its own joints out instead of guessing. YOLOv8's is
Missing-Detection — no person, no pose — and it has the largest failure count
overall.

So the pattern is: commit-then-predict. Graceful degradation. All-or-nothing. Each
signature matches the design logic of its paradigm. I report this as an
observational correspondence — with one model family per paradigm, I cannot
separate architecture from training data. But as a hypothesis, it is consistent.

---

## Folie 21 — The COCO check (1:00)

One final check before the synthesis. REHAB24-6 was my main benchmark. But would a
standard benchmark give the same answer? To test that, all nine variants also ran on
COCO — as an auxiliary comparison, accuracy only.

And the answer is: only partially. [Slopegraph] Look at MoveNet MultiPose, the winner
on REHAB — on COCO it falls back to rank four. The new leader there is YOLOv8 Medium,
and MediaPipe falls even further. If we only look at the three main models, the
picture is calmer: MoveNet stays in front on both datasets, and MediaPipe and YOLOv8
simply swap places.

Now, why am I careful with this conclusion? Because the two evaluations don't just
use different data — they also follow different rules. On COCO there is no patient.
So I cannot let the pipeline pick the most prominent person — I have to match each
prediction to the person that was annotated. And that means: when the rankings shift,
I cannot say how much of that comes from the new domain, and how much from the
different rules. So I keep the claim modest: rankings transfer only partially. But
even that has a practical message — a COCO leaderboard alone would not tell you which
model to deploy for rehabilitation.

---

## Folie 22 — Three profiles (1:45)

So — back to the original question: which model for home rehabilitation?

The honest answer of six dimensions: there is no single winner. There are three
architectural profiles — one for each of the three paradigms from the beginning.
[Spalten] MediaPipe, the filtered top-down profile: safest
around the coach, richest landmarks — but slower, and weak hips. MoveNet, the
low-exposure bottom-up profile: best accuracy, stability, and speed — but the least
complete skeletons. YOLOv8, the high-sensitivity one-stage profile: the most complete
detections — but it sees other people most often, and it is the most
viewpoint-sensitive.

Which model is right depends on which dimension your application cannot afford to
lose.

---

## Folie 23 — Decision guide (1:00)

As practical guidance: for body-joint analysis on CPU, MoveNet MultiPose is the
default choice. If you need the complete skeleton, take YOLOv8. If coach robustness
or the richer landmarks matter, take MediaPipe. If only speed counts, MoveNet
SinglePose Lightning runs at 100 FPS.

And one insight beyond the model choice: guide the patient to a frontal camera
position, watch for second persons, and warn when predictions get unstable. These
deployment decisions matter as much as the model itself. And with that, all five
questions from the beginning have an answer.

---

## Folie 24 — Limitations (1:15)

Of course, these results have clear boundaries. And each boundary points to a next
step.

The most important one: my subjects were ten healthy adults, in motion-capture suits.
Real patients look different and move differently. They are older. Different body
shapes. Everyday clothing. Sometimes a walking aid. And compensatory movements. All
of that can shift pose accuracy itself. So the first item of future work is
validation with real patients.

Second: I evaluated raw outputs, frame by frame. That is an upper bound on failures —
a real system with smoothing would typically look better, though tracking brings its
own failure modes. The follow-up: apply the same smoothing to all models. Third: the COCO comparison used a different protocol — a
standardized protocol would sharpen it. And fourth: this is 2D. Real assessment
ultimately needs 3D joint angles.

These are not oversights. They are scope decisions — and each one defines the next
experiment.

---

## Folie 25 — Conclusion (0:45)

Two findings to take home. First: on a rehabilitation benchmark with
motion-capture ground truth, MoveNet MultiPose is the strongest CPU default — most
accurate, most stable, fastest of the main models. Second: the three models do
not form a ranking. They form three architectural profiles — detection count does not
explain coach robustness, and each architecture fails in its own characteristic way.

---

## Folie 26 — Thank you (0:15 + Closer)

So — can you trust a mobile pose estimator in a living room? Yes. But which one
depends on what your application cannot afford to lose. MoveNet MultiPose is the
default. The three architectures fail in three different ways — and knowing those
ways is as valuable as any ranking. The next step toward the clinic is validation on
real patients. Thank you.

---

# Lern-Anleitung (FINAL v5 — Deck ist eingefroren)

1. Neu seit Deinem letzten Run: Folie 5 (Scope — die Umfangs-Zahlen) und der
   Familien-Absatz auf Folie 14 (Speed). Beide je einmal üben.
2. Tempo halten wie im letzten Run (der war richtig): Pausen nach Titeln,
   Signposts (Ende Folie 9 + 14) absetzen, auf Figuren zeigen.
3. Live-Steuerung: Zu lang? Folie 19 skippen (ein Klick). Rubber: 12, 21, 23.
   NIE kürzen: 18 (Dreisatz), 22 (Profile).
4. Ziel-Korridor: 25–29 Min gesprochen. Alles darin = perfekt. Keine weiteren
   Änderungen mehr — ab jetzt nur noch üben und schlafen.
