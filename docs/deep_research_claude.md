# Choosing a second dataset for rehabilitation pose estimation benchmarking

**KIMORE is the strongest complementary dataset for your thesis**, offering RGB video of real patients doing rehabilitation exercises with projectable 3D skeleton ground truth — though no candidate perfectly matches REHAB24-6's optical motion capture quality. Of the 7 specifically requested datasets and 7 additional candidates evaluated, only 3 meet the minimum requirements of RGB video plus usable keypoint ground truth in a clinical or exercise context. Most rehabilitation datasets provide skeleton-only or depth-only data without RGB, making them incompatible with evaluating camera-based 2D pose estimators like MediaPipe, MoveNet, and YOLOv8-Pose.

---

## Detailed evaluation of each requested dataset

### 1. KIMORE — the leading candidate with caveats

**Full name:** KIMORE — KInematic Assessment of MOvement and Clinical Scores for Remote Monitoring of Physical REhabilitation. Capecci et al. (2019), *IEEE Trans. Neural Systems and Rehabilitation Engineering*, 27(7), 1436–1448.

**Format:** RGB video at **1920×1080** from Kinect V2 (30 fps), plus depth video (512×424), plus 25-joint 3D skeleton with positions and quaternion orientations. All three modalities are provided per recording.

**Annotation scheme:** 25 Kinect V2 joints in 3D camera coordinates (cameraX, cameraY, cameraZ) with confidence states. **All 12 required joints are directly available** — shoulders, elbows, wrists, hips, knees, ankles map straightforwardly. Not natively COCO-17, but 13–14 joints map cleanly (eyes and ears are absent from Kinect). Clinical quality scores from physicians are also provided per exercise.

**Scale:** **78 subjects** (44 healthy controls + 34 patients with motor dysfunctions including back pain, stroke, and Parkinson's disease), performing **5 low back pain rehabilitation exercises** (arm lifting, lateral trunk tilt, trunk rotation, pelvis rotation, squatting). Approximately **390 total sequences**.

**Download:** Free via SharePoint link from the VRAI lab page at https://vrai.dii.univpm.it/content/kimore-dataset. No lengthy approval — download is essentially instant once the EULA is accepted.

**Compatibility verdict: YES, with projection work.** You can extract RGB frames and run all three pose estimators on them. The critical step is projecting the 3D Kinect skeleton to 2D pixel coordinates on the RGB image. **Camera intrinsic parameters are not provided in the dataset.** However, Kinect V2 has well-known approximate intrinsics (fx≈fy≈1081, cx≈960, cy≈540 for 1920×1080 RGB), and using these introduces only a few pixels of error. No published paper has explicitly performed this 3D→2D projection for HPE accuracy benchmarking on KIMORE, though multiple papers have run OpenPose and BlazePose on the RGB frames for rehabilitation assessment comparison (Marušić et al., 2024, arXiv:2408.02855).

**Recommendation tier: STRONG.** The combination of real patients, clinical rehabilitation context, RGB video, and free instant access makes KIMORE the best overall match. The 3D→2D projection requirement is a manageable engineering task, and the Kinect V2 skeleton — while less accurate than optical mocap — is an accepted ground truth standard in the rehabilitation literature.

---

### 2. UI-PRMD — skeleton-only, no RGB

**Full name:** University of Idaho Physical Rehabilitation Movement Data. Vakanski et al. (2018), *Data*, 3(1), 2.

**Format:** Skeleton data only — 3D joint positions and Euler angles from both Vicon (39 joints, 100 Hz) and Kinect (22 joints, 30 Hz). **No RGB video, no depth maps, no images of any kind.**

**Scale:** 10 healthy subjects (no patients), 10 exercises, 2,000 sequences (10 correct + 10 incorrect per exercise per subject).

**Download:** Instant, free at http://webpages.uidaho.edu/ui-prmd/ (Open Data Commons Public Domain).

**Compatibility verdict: NO.** Without any RGB data, it is impossible to run MediaPipe, MoveNet, or YOLOv8-Pose. This dataset is designed for movement quality classification from skeleton data, not for pose estimation benchmarking.

**Recommendation tier: NOT SUITABLE.**

---

### 3. IntelliRehabDS (IRDS) — depth and skeleton only, no RGB

**Full name:** IntelliRehabDS — A Dataset of Physical Rehabilitation Movements. Miron et al. (2021), *Data*, 6(5), 46.

**Format:** 25-joint Kinect V2 3D skeleton (CSV) + raw depth map images. **No RGB video.** The "raw" CSV format does include 2D projections of the 3D skeleton, but these correspond to depth frames, not RGB images.

**Scale:** **29 subjects** (15 patients with stroke, spinal cord injury, brain injury, fractures, amputation + 14 healthy controls), **9 upper-body exercises**, 2,577 usable sequences. Strong clinical relevance with real patients in a rehabilitation centre.

**Download:** Instant, free via Zenodo at https://zenodo.org/records/4610859 (CC-BY-4.0).

**Compatibility verdict: NO.** Despite excellent clinical content and real patients, the absence of RGB video is a dealbreaker. Running RGB-trained pose estimators on depth maps produces meaningless results — the visual appearance is fundamentally different.

**Recommendation tier: NOT SUITABLE.**

---

### 4. UCO Physical Rehabilitation — RGB with 2D ground truth, but only 3 joints per video

**Full name:** UCO Physical Rehabilitation (ucophyrehab). Aguilar-Ortega et al. (2023), *Sensors*, 23(21), 8862. Note: the name "UCO-MPOSE" does not appear in the literature — this is the actual dataset.

**Format:** RGB video at **1280×720** from **5 synchronized cameras** at 3 heights and 3 angles (AVI format), plus 3D ground truth from a 6-camera **OptiTrack IR motion capture system** (Flex 3) with reflective markers, plus **2D projected ground truth** (cam*_p2d.txt files and dataset_2d.json).

**Annotation scheme:** This is the critical limitation. OptiTrack markers were placed on only **3 joints per exercise set**: upper-body exercises annotate shoulder, elbow, wrist; lower-body exercises annotate hip, knee, ankle. **You never get more than 3 joints simultaneously in any single video.** This means a full 12-joint evaluation is impossible per-video; you would need to aggregate across exercise types.

**Scale:** 27 subjects (ages 23–60), 16 exercises (8 upper + 8 lower body, left and right), ~2,160 video sequences across 5 camera views. Average video duration ~30 seconds, totaling ~1.6 million frames.

**Download:** GitHub at https://github.com/AVAuco/ucophyrehab, but **access requires email request** to inforeha@uco.es with name, affiliation, and purpose. Approval timeline is unspecified but likely days.

**Compatibility verdict: PARTIAL.** You can run all three pose estimators on the RGB frames and compare against real OptiTrack-based 2D ground truth — the gold-standard pipeline. But the 3-joint-per-video limitation severely restricts the evaluation. The original paper itself uses this approach to benchmark HPE models including MediaPipe and YOLOv8-Pose, so methodology is validated. Multi-view capability is a plus.

**Recommendation tier: POSSIBLE.** High-quality ground truth (OptiTrack) and the paper already benchmarks the exact models you use. But the sparse joint annotation and email access process are drawbacks.

---

### 5. Fitness-AQA — no keypoint annotations at all

**Full name:** Fitness-AQA. Parmar, Gharat, & Rhodin (2022), *ECCV 2022*, pp. 105–123.

**Format:** RGB video clips scraped from YouTube and Instagram (~4 seconds each). Variable resolution and quality.

**Annotation scheme:** **No 2D or 3D keypoint annotations whatsoever.** Only binary exercise error labels (e.g., "Knees Inward Error," "Shallow Squat") annotated by professional gym trainers. Designed for action quality assessment, not pose estimation.

**Scale:** ~21,284 samples across BackSquat, BarbellRow, and OverheadPress. Subject count unknown (crowdsourced from social media).

**Download:** Via Google Form at https://forms.gle/PbPTX1eVxGpa3QG88 and Zenodo at https://zenodo.org/records/7310289. Non-commercial license.

**Compatibility verdict: NO.** Without any keypoint ground truth, there is nothing to evaluate pose estimator accuracy against.

**Recommendation tier: NOT SUITABLE.**

---

### 6. NTU RGB+D 120 — massive but not rehabilitation-focused

**Full name:** NTU RGB+D 120. Liu et al. (2020), *IEEE TPAMI*, 42(10), 2684–2701 (original NTU RGB+D 60: Shahroudy et al., CVPR 2016).

**Format:** RGB video at **1920×1080**, depth maps (512×424), IR video (512×424), and 25-joint Kinect V2 3D skeleton. All modalities available per sample. Modalities downloadable separately.

**Annotation scheme:** 25 Kinect V2 joints with 3D positions. MATLAB mapping code is provided to project 3D skeletons to RGB pixel coordinates. A third-party resource (NTU_motion_sim_annotations on GitHub) provides refined 2D annotations in **COCO JSON format** generated by MultiPoseNet. All 12 required joints are available.

**Scale:** **106 subjects**, **114,480 video samples**, ~8 million frames, **120 action classes** across 96 backgrounds. By far the largest dataset evaluated.

**Rehabilitation relevance is weak.** The 120 classes include 12 "health-related" actions, but these are **symptom presentations** (staggering, falling, touching head/chest/back/neck) rather than therapeutic exercises. No squats, no arm lifts, no trunk rotations as rehabilitative movements. This makes it a poor thematic match for a rehabilitation-focused thesis.

**Download:** Registration + approval at https://rose1.ntu.edu.sg/dataset/actionRecognition/. Approval is manual but typically completes in **a few business days**. Skeleton-only data is available instantly via Google Drive.

**Compatibility verdict: PARTIAL.** Technically feasible — RGB video exists, 3D→2D projection tools are provided, and COCO-format 2D annotations exist from third parties. But the lack of rehabilitation exercises undermines the cross-dataset generalization argument for a rehab thesis.

**Recommendation tier: POSSIBLE** (for demonstrating generalization to non-clinical movement, but weak thematic alignment).

---

### 7. Physio2.2M — ideal format but not publicly available

**Full name:** Physio2.2M. Rode et al. (2025), "Assessment of monocular human pose estimation models for clinical movement analysis," *Scientific Reports*, 15, 38767.

**Format:** RGB video at 30 fps synchronized with **27-camera Vicon motion capture** at 200 Hz. 46 anatomical passive markers provide gold-standard 3D ground truth. Multiple camera viewing angles included. **~2.2 million RGB frames total.**

**Annotation scheme:** 46 markers projectable to any standard skeleton format. The paper itself benchmarks COCO-17 models (MoveNet, MediaPipe/BlazePose, YOLO-based), confirming mapping to standard formats. Evaluates knee and elbow flexion angles. All 12 required joints covered.

**Scale:** 25 unimpaired participants (11 male, 14 female, ages 20–33), 7 physiotherapy exercises (squats, bridge, bird/quadruped, abduction, shoulder exercises, knee exercises, stretches), multiple camera angles.

**Download:** **Not publicly available.** Described as an internal dataset of **Akina AG** (Swiss health-tech company). Must contact corresponding author David Rode at ETH Zurich. Given the corporate ownership, access may be refused or significantly delayed.

**Compatibility verdict: YES in principle** — this is exactly the right format. The paper does precisely what your thesis does. But access is the blocker.

**Recommendation tier: NOT SUITABLE** due to access constraints, despite being technically ideal.

---

## Additional candidate datasets evaluated

### Keraal — model-estimated "ground truth" disqualifies it

Nguyen et al. (2024), arXiv:2407.00521. 21 subjects (12 low-back-pain patients + 9 healthy), 3 exercises, 2,622 recordings. RGB video at 480×360 (Groups 1–2) and 960×544 (Group 3). Free instant download at https://keraal.enstb.org/KeraalDataset.html under CC-BY-NC-SA.

**The critical problem:** the provided 2D keypoints (OpenPose COCO format, BlazePose 33 joints) are **model-estimated from the RGB video, not gold-standard annotations**. The paper explicitly states these were obtained "by post-processing of the RGB video." Vicon mocap exists only for Group 3 (540 recordings, healthy subjects only), with no camera projection matrices provided. **Comparing one pose estimator's output against another's is circular** and cannot serve as a proper benchmark. Additionally, the video resolution (480×360) is very low.

**Recommendation tier: NOT SUITABLE** for HPE accuracy benchmarking.

### MobiPhysio — no keypoint annotations

Iqbal et al. (2026), *Data in Brief*, 65, 112635. 58 participants, 9 physiotherapy exercises, 3,686 videos recorded from smartphones at 1080p/30fps from 3 camera angles. Includes lighting/jitter/occlusion variations. CC-BY-4.0 license.

**Provides only exercise quality assessment scores (EAAQ), not keypoint annotations.** Strong as a test set for running pose estimators in realistic smartphone conditions, but with no ground truth for accuracy evaluation.

**Recommendation tier: NOT SUITABLE** for benchmarking (but useful for qualitative analysis).

### Fit3D — high-quality ground truth, fitness context

Fieraru et al. (2021), "AIFit," *CVPR 2021*, pp. 9919–9928. 13 subjects, **611 multi-view sequences**, 37 exercises, **>3 million frames**. 4 synchronized RGB cameras + 12-camera **Vicon mocap** (sub-millimeter accuracy). Provides GHUM and SMPL-X body model parameters, calibrated camera intrinsics/extrinsics enabling reliable 3D→2D projection. Published work (BlanketGen2-Fit3D) has evaluated **PCK@0.05** on Fit3D, confirming 2D benchmarking feasibility.

**Download requires registration and approval** at https://fit3d.imar.ro/download. Timeline unspecified but typically days to weeks.

**Recommendation tier: POSSIBLE.** Excellent technical quality and multi-view setup. The ground truth is the most reliable of any evaluated dataset. However, it covers **gym/fitness exercises** (not clinical rehabilitation), has only 13 subjects, and requires approval.

### InfiniteRep — synthetic but perfect COCO annotations

Infinity AI (2021). 1,000 synthetic RGB videos at 224×224 (24 fps), 10 exercises (squats, push-ups, shoulder press, etc.). **Pixel-perfect COCO-17 2D keypoint annotations.** CC-BY-4.0, free download at https://marketplace.infinity.ai/pages/infiniterep-dataset.

The annotations are flawless by construction, but **synthetic avatars do not generalize to real patients.** The low resolution (224×224) further limits utility.

**Recommendation tier: NOT SUITABLE** for a clinical generalization study (useful only for controlled methodology validation).

### Datasets confirmed as nonexistent or irrelevant

**ExerciseNet, MedPose, KInD, REHABOT, HA4D:** No public datasets by these names were found in academic literature or dataset repositories. **LOVEU:** A video understanding challenge, not rehabilitation-specific. **SPHERE:** Smart-home sensors, no standard RGB pose data. **EmoPain:** Chronic pain dataset with restricted access and designed for pain detection, not HPE benchmarking. **FineGym:** Competitive gymnastics with no skeleton annotations. **Skeletics-152:** Model-estimated poses from Kinetics-700, not gold standard. **PKU-MMD:** Similar to NTU RGB+D but fewer actions and no rehabilitation relevance. **MM-Fit:** Model-estimated 2D poses only, not gold standard. **M3GYM:** CVPR 2025, availability uncertain as of April 2026.

---

## Why no dataset perfectly matches REHAB24-6

Your primary dataset, REHAB24-6, occupies a rare niche: **optical motion capture ground truth projected to 2D pixel coordinates on RGB video of real rehabilitation patients**. This combination is exceptionally uncommon in public datasets. Most rehabilitation datasets either lack RGB video (UI-PRMD, IntelliRehabDS), lack gold-standard annotations (Keraal, MobiPhysio), or lack clinical context (NTU RGB+D, Fit3D). The closest match — Physio2.2M — exists but is locked behind corporate access. This gap in the field is itself a contribution worth noting in your thesis.

---

## Ranked shortlist: the top 3 candidates

**Rank 1: KIMORE** — The clear best choice. It provides RGB video of **real patients** (34 with motor dysfunctions) doing **5 clinical rehabilitation exercises** with physician-scored quality assessments. The 25-joint Kinect V2 skeleton covers all 12 required joints and can be projected to 2D using known approximate intrinsics. Free instant download. The main limitation — using approximate camera intrinsics for 3D→2D projection — introduces modest error (estimated few-pixel range) but is a well-understood trade-off. KIMORE is the most-cited rehabilitation exercise dataset in the HPE literature, and published work has already applied OpenPose and BlazePose to its RGB frames. **This dataset most closely mirrors REHAB24-6's clinical rehabilitation context while adding a different sensor modality (Kinect vs. optical mocap) for genuine cross-dataset generalization testing.**

**Rank 2: Fit3D** — The strongest technical option if rehabilitation context is flexible. Vicon mocap with calibrated multi-view RGB cameras provides the most reliable 2D ground truth of any evaluated dataset. The 37 exercises include several rehab-adjacent movements (squats, bridges, stretches). The multi-view setup enables viewpoint-dependence analysis. The drawbacks are the approval-gated access, only 13 subjects, and fitness (not clinical) framing.

**Rank 3: UCO Physical Rehabilitation** — Provides the only dataset with true optical-mocap-based 2D projected ground truth in a rehabilitation context and multi-camera setup. The original paper explicitly benchmarks MediaPipe and YOLOv8-Pose, making methodology directly comparable to your thesis. However, the **3-joint-per-video annotation** severely limits evaluation scope, and the email access request adds friction. Best used as a supplementary validation if combined with KIMORE.

For a bachelor thesis seeking one additional dataset, **KIMORE offers the best balance** of rehabilitation relevance, data quality, accessibility, and established use in the literature. The 3D→2D projection step is additional engineering work, but it is tractable and adds methodological depth to the thesis by demonstrating how Kinect-based ground truth compares to optical motion capture.