Perplexity:
For your specific constraints, the **best fit is KERAAL (low‑back‑pain rehab)** as a second dataset, with **UCO Physical Rehabilitation** a strong additional option if you accept a short e‑mail access request and limb‑only ground truth. KIMORE/UI‑PRMD/IntelliRehabDS/Fitness‑AQA/NTU RGB+D/Physio2.2M all miss at least one of your hard requirements (RGB video, ready‑to‑use 2D joints, or open download) for your cross‑dataset HPE evaluation. [github](https://github.com/petteriTeikari/KiMoRe_wrapper)

Below I go through (1) the two strongest candidates, then (2) each dataset you listed, with format, joints, size, download, and suitability.

***

## Quick suitability overview

| Dataset | RGB video | 2D joints provided | Rehab / physio context | Open download (no approval) | Comment for your thesis |
|--------|-----------|--------------------|------------------------|-----------------------------|-------------------------|
| **KERAAL** | Yes | Yes (COCO‑style/OpenPose, plus BlazePose 33‑keypoint) | Clinical low‑back‑pain rehab | Yes (direct link) | Best match; full‑body 2D, real patients, rehab labels. [github](https://github.com/nguyensmai/KeraalDataset) |
| **UCO Physical Rehab** | Yes | Yes (projected from OptiTrack) | Rehab exercises, clinic | Email request | Excellent limb‑level GT; only 3 joints per limb; access via email. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/) |
| **KIMORE** | Yes | No (3D Kinect only) | Back‑pain rehab | Yes | You’d have to derive 2D from Kinect; no ready COCO‑style 2D. [vrai.dii.univpm](https://vrai.dii.univpm.it/content/KiMoRe-dataset) |
| **UI‑PRMD** | No RGB (skeleton only) | No (3D joints & angles only) | Rehab exercises | Yes | Good for skeleton‑based work, but you can’t run MediaPipe/MoveNet/YOLO on RGB. [selectdataset](https://www.selectdataset.com/dataset/7c1dfcb8e97bb3483b972e29f966969a) |
| **IntelliRehabDS** | No RGB (depth + 3D skeleton) | No | Mixed rehab/controls | Yes | Kinect‑only; fails your RGB requirement. [zenodo](https://zenodo.org/records/4610859) |
| **Fitness‑AQA** | Yes | No pose GT (error labels only) | Gym fitness | Yes | Great for AQA, but no 2D coordinates. [arxiv](https://arxiv.org/pdf/2202.14019.pdf) |
| **NTU RGB+D** | Yes | Effectively yes (3D + pixel indices ⇒ 2D) | General actions, a few medical signs | Request + release agreement | Huge, but not rehab‑specific and has a formal access process. [emergentmind](https://www.emergentmind.com/topics/ntu-rgb-d-120-dataset) |
| **Physio2.2M** | Yes | Internally 3D MoCap; public data not open | Physical exercise (healthy) | Not publicly available | Data only “available from corresponding author on reasonable request.” [github](https://github.com/petteriTeikari/KiMoRe_wrapper) |

***

## Strongest candidates for your cross‑dataset evaluation

### 1. KERAAL: low‑back‑pain rehab (recommended)

**Context and modalities**

- Clinical dataset from a prospective trial on **chronic low‑back‑pain rehabilitation**, with 3 stretching/rehab exercises coached by a humanoid robot. [github](https://github.com/nguyensmai/KeraalDataset)
- **Participants:** 9 healthy subjects + 12 low‑back‑pain patients (21 total). [github](https://github.com/nguyensmai/KeraalDataset)
- **Recordings:** Total **2622 exercise recordings** across five groups (1a,1b,2a,2b,3). [themoonlight](https://www.themoonlight.io/tw/review/a-medical-low-back-pain-physical-rehabilitation-dataset-for-human-body-movement-analysis)
- **Data types per repetition:**  
  - RGB video (mp4 480×360 for Groups 1/2, avi 960×544 for Group 3). [github](https://github.com/nguyensmai/KeraalDataset)
  - Kinect V2 skeleton: 3D positions + orientations for each joint (25‑joint Kinect model). [github](https://github.com/nguyensmai/KeraalDataset)
  - Vicon skeleton (3D MoCap) for Group 3 recordings. [themoonlight](https://www.themoonlight.io/tw/review/a-medical-low-back-pain-physical-rehabilitation-dataset-for-human-body-movement-analysis)
  - 2D skeletons estimated from RGB with **OpenPose (COCO pose format)** and **BlazePose (33 landmarks)**. [arxiv](https://arxiv.org/html/2407.00521v2)
  - Rich clinical annotations (correct/incorrect, error labels, body part, time span) from a physiatrist. [themoonlight](https://www.themoonlight.io/tw/review/a-medical-low-back-pain-physical-rehabilitation-dataset-for-human-body-movement-analysis)

**Joint annotation scheme**

- **OpenPose 2D skeleton:** joints in **COCO pose output format**, i.e. standard COCO body keypoints (17/18 joints) with x,y per frame. [arxiv](https://arxiv.org/html/2407.00521v2)
- **BlazePose 3D skeleton:** 33 keypoints (BlazePose whole‑body topology, superset of COCO/BlazeFace/BlazePalm). [arxiv](https://arxiv.org/html/2407.00521v2)
- **Kinect skeleton:** 25 3D joints (Kinect V2 convention); plus positions/orientations. [github](https://github.com/nguyensmai/KeraalDataset)
- **Vicon skeleton:** 3D joint positions for a subset (Group 3), giving you a higher‑accuracy reference for some sequences. [themoonlight](https://www.themoonlight.io/tw/review/a-medical-low-back-pain-physical-rehabilitation-dataset-for-human-body-movement-analysis)

You can map COCO/BlazePose easily to your 12‑joint subset (L/R shoulder, elbow, wrist, hip, knee, ankle).

**Number of subjects/videos**

- 21 subjects (9 healthy, 12 patients) in a clinical low‑back‑pain program. [themoonlight](https://www.themoonlight.io/tw/review/a-medical-low-back-pain-physical-rehabilitation-dataset-for-human-body-movement-analysis)
- **Total 2622 recordings** across patient and healthy groups; three exercises. [github](https://github.com/nguyensmai/KeraalDataset)

**How to download**

- Dataset and pretrained OpenPose model are linked from the project page and GitHub; the GitHub README points to `http://nguyensmai.free.fr/KeraalDataset.html` as the download entry. [github](https://github.com/nguyensmai/KeraalDataset)
- No institutional approval; you download zipped folders (RGB, Kinect, Vicon, OpenPose, BlazePose, annotations) directly once you accept the license terms. [github](https://github.com/nguyensmai/KeraalDataset)

**Compatibility with MediaPipe / MoveNet / YOLOv8‑Pose**

- **Running models:** RGB videos are standard compressed streams (480×360 or 960×544), so you can run **MediaPipe Pose/BlazePose, MoveNet, and YOLOv8‑Pose** frame‑by‑frame with no sensor‑specific issues. [github](https://github.com/nguyensmai/KeraalDataset)
- **2D “ground truth”:**  
  - You get ready‑made **COCO‑style 2D keypoints from OpenPose** for all frames. [github](https://github.com/nguyensmai/KeraalDataset)
  - You can treat these as a **strong pseudo‑ground‑truth** baseline for 2D HPE, or instead derive 2D by projecting Kinect/Vicon skeletons into the image plane if you are willing to calibrate. [arxiv](https://arxiv.org/html/2407.00521v2)
- **Pros for your thesis:**  
  - Real rehab patients, clinical protocol, medically annotated errors → very strong fit to your home‑rehab framing. [themoonlight](https://www.themoonlight.io/tw/review/a-medical-low-back-pain-physical-rehabilitation-dataset-for-human-body-movement-analysis)
  - Full‑body 2D skeleton for many joints (COCO / 33‑keypoint) → matches your “≥12 joints” requirement.  
- **Limitations:**  
  - The only **strict “ground truth”** is Kinect/Vicon 3D; the provided 2D skeletons are estimates from OpenPose/BlazePose, so if you use them as GT you are evaluating your models against other HPE algorithms, not independent MoCap. You can mitigate this by:
    - Restricting some analyses to the **Vicon‑recorded subset**, where you can triangulate and then project to 2D for higher‑fidelity GT.  
    - Explicitly labeling the 2D skeleton as “reference model” rather than “ground truth” in your thesis.  

Overall, KERAAL checks all the hard boxes except that its 2D keypoints come from HPE models rather than manual/MoCap labeling. In return you get real patients, clinical rehab context, standard COCO keypoints, and frictionless download.

***

### 2. UCO Physical Rehabilitation (UCO‑PhyRehab / “UCO‑MPOSE”)

**Context and modalities**

- Dataset introduced in *“UCO Physical Rehabilitation: New Dataset and Study of Human Pose Estimation Methods on Physical Rehabilitation Exercises”* (Sensors 2023). [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)
- **Participants:** 27 healthy subjects (7 female, 20 male), age 23–60. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)
- Exercises are typical **post‑surgery rehab movements** for knee and shoulder, in seated, standing and supine positions, designed by surgeons/physiotherapists. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)
- **Recordings:** 16 exercise types, **2160 RGB video sequences** (~1.6M frames), each ~30.4 s, captured simultaneously by **5 RGB cameras (1280×720)** at different heights and viewpoints. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)
- Simultaneous **OptiTrack IR motion‑capture** with markers on hip/knee/ankle or shoulder/elbow/wrist, depending on exercise. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)

**Joint annotation scheme (2D + 3D)**

- For each frame they store:
  - 3D positions for 3 joints per limb (shoulder–elbow–wrist for upper‑limb exercises; hip–knee–ankle for lower‑limb). [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)
  - **2D ground‑truth joint positions (per camera)** obtained by projecting the OptiTrack 3D joints using calibrated camera intrinsics/extrinsics. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)
  - Missing 3D points are manually **filled in 2D** where possible (for some frames OptiTrack lost a marker; these frames have 2D but no 3D GT). [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)
- Joint order:  
  - Upper body exercises: index 0–2 = shoulder, elbow, wrist.  
  - Lower body exercises: index 0–2 = hip, knee, ankle. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)

This is **high‑quality, MoCap‑based 2D ground truth**, but **only for three joints per exercise**.

**Number of subjects/videos**

- 27 subjects × 16 exercises × 4 repetitions × 5 cameras → **2160 video sequences**, ~1.6M frames. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)

**How to download**

- GitHub repository `AVAuco/ucophyrehab` documents the structure; 2D and 3D annotations are stored in `dataset_2d.json`, `dataset_3d.json`, plus `camX.avi` and `camX_p2d.txt` files. [github](https://github.com/AVAuco/ucophyrehab)
- **Access:** the authors ask you to email `inforeha@uco.es` with your name, affiliation and research purpose. [github](https://github.com/AVAuco/ucophyrehab)
  - This is a lightweight manual step (not a formal ethics board portal), but still technically violates your “no access requests” preference.  

**Compatibility with MediaPipe / MoveNet / YOLOv8‑Pose**

- **Running models:** straightforward — standard 1280×720 RGB sequences from 5 viewpoints, so all three HPE models will work out of the box on each camera stream. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)
- **Comparing to 2D GT:**  
  - You can compute per‑frame errors for the 3 tracked joints (hip, knee, ankle or shoulder, elbow, wrist) per camera.  
  - This supports a **clean limb‑focused evaluation** (e.g., knee angle accuracy during flexion) but **not a full 12‑joint skeleton evaluation**, because no torso / head / other limb joints are annotated.  
- **Pros for your thesis:**  
  - Very close to your own dataset concept: rehab exercises, multiple camera views, synchronized MoCap → 2D GT, plus an existing benchmark paper on HPE in rehab. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)
- **Limitations:**  
  - Healthy volunteers only (no patients), but with slowed, “post‑surgery–like” execution. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)
  - Only 3 joints per limb, so your cross‑dataset section would have to be explicitly **“limb‑level generalization on hip/knee/ankle and shoulder/elbow/wrist”** instead of full‑body.  
  - Small email‑based access friction.  

If you’re okay with evaluating a **subset of joints**, UCO is an excellent second dataset because it gives you **true MoCap‑derived 2D ground truth** in a rehab setting and 5 camera viewpoints per recording.

***

## Evaluation of the other datasets you listed

### KIMORE

- **Format / context**  
  - KInematic Assessment of MOvement and Clinical Scores for Remote Monitoring of Physical REhabilitation; rehab exercises for **low back pain**. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/31217121/)
  - **Sensors:** Kinect V2 (RGB‑D) in a lab; they recorded **RGB, depth, and Kinect skeleton joint positions** during 5 rehab exercises selected by physicians. [vrai.dii.univpm](https://vrai.dii.univpm.it/content/KiMoRe-dataset)
  - **Population:** 78 subjects (44 controls, 34 patients with motor dysfunction). [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/31217121/)
- **Joints & annotations**  
  - Kinect provides **3D positions and orientations of 25 joints**; dataset also includes physiotherapist‑defined kinematic features and clinical scores per trial. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0010482524016639)
  - There is **no explicit 2D joint coordinate file in image pixels**; you only get Kinect 3D coordinates, not COCO‑style 2D keypoints. [vrai.dii.univpm](https://vrai.dii.univpm.it/content/KiMoRe-dataset)
- **Download**  
  - Publicly downloadable from the authors’ site; the IEEE paper links a SharePoint where you can directly fetch the data. [vrai.dii.univpm](https://vrai.dii.univpm.it/content/KiMoRe-dataset)
- **Suitability**  
  - Good for **3D skeleton–based rehab assessment**, but for your use case:
    - You’d need to **reproject Kinect 3D joints into RGB coordinates**, which requires camera intrinsics/extrinsics and will still give you Kinect’s skeleton as “ground truth”, not MoCap.  
    - No COCO‑like 2D keypoint set is provided; mapping to your 12 joints is possible but requires additional work.  
  - Compatible with running MediaPipe/MoveNet/YOLO on RGB, but **does not meet your requirement of ready‑to‑use 2D ground truth**.

### UI‑PRMD (University of Idaho – Physical Rehabilitation Movement Dataset)

- **Format / context**  
  - 10 common rehab movements (e.g. squats, lunges, upper‑limb tasks); 10 healthy subjects, each repeating motions 10 times. [selectdataset](https://www.selectdataset.com/dataset/7c1dfcb8e97bb3483b972e29f966969a)
  - Recorded simultaneously with **Vicon optical MoCap** and **Kinect**, but the released dataset focuses on **joint angles and positions**, not raw RGB. [pdfs.semanticscholar](https://pdfs.semanticscholar.org/b1ee/90f1aa9b380f5400b233dee2e4f0a33e71c7.pdf)
- **Joints & annotations**  
  - Data is provided as **3D joint positions and joint angles** (Vicon and Kinect skeletal models), e.g. 18–25 joints in 3D. [arxiv](https://arxiv.org/html/2507.21018)
  - No per‑frame **2D pixel coordinates** and generally **no RGB videos** in the public release; most downstream work uses skeleton sequences only. [selectdataset](https://www.selectdataset.com/dataset/7c1dfcb8e97bb3483b972e29f966969a)
- **Download**  
  - Publicly downloadable from repositories like OpenDataLab and the authors’ site (no approval step). [opendatalab](https://opendatalab.com/OpenDataLab/UI-PRMD/download)
- **Suitability**  
  - Excellent for skeleton‑based rehab assessment and quality scoring, but:
    - You **cannot run MediaPipe/MoveNet/YOLO on RGB for evaluation**, because RGB streams are not provided.  
    - You would have to derive 2D coordinates from 3D for each joint, and it’s still non‑RGB‑centric.  
  - So UI‑PRMD does **not satisfy your RGB‑video + 2D GT requirement**.

### IntelliRehabDS

- **Format / context**  
  - Kinect‑based dataset of 9 rehabilitation gestures (sitting/standing upper & lower limb movements) from **29 subjects** (15 patients, 14 healthy controls). [bura.brunel.ac](https://bura.brunel.ac.uk/handle/2438/24189)
- **Joints & annotations**  
  - Provides **3D coordinates of 25 body joints** (Kinect skeleton) and **depth maps**; no RGB color frames. [zenodo](https://zenodo.org/records/4610859)
  - Each repetition is labeled with gesture type, sitting/standing, and correctness (correct vs incorrect). [bura.brunel.ac](https://bura.brunel.ac.uk/handle/2438/24189)
- **Download**  
  - Freely downloadable from Zenodo; GitHub helper code is public. [github](https://github.com/alina-miron/intellirehabds)
- **Suitability**  
  - Fails two of your core constraints:
    - No **RGB video** (depth only).  
    - No **2D pixel keypoints**; only 3D Kinect skeleton.  
  - So it is **not suitable** as the second dataset for your RGB‑based HPE evaluation.

### Fitness‑AQA

- **Format / context**  
  - ECCV 2022 dataset for **in‑the‑wild fitness Action Quality Assessment**, with real gym videos of three exercises: Back Squat, Barbell Row, Overhead Press. [arxiv](https://arxiv.org/pdf/2202.14019.pdf)
  - RGB videos (and selected frames) collected from YouTube/Instagram; annotated by expert trainers for fine‑grained error types and quality scores. [arxiv](https://arxiv.org/pdf/2202.14019.pdf)
- **Joints & annotations**  
  - The authors deliberately **do not provide pose annotations**; they explicitly state that the labeled dataset “contains only the exercise error as ground‑truth annotation and no information related to human body pose.” [arxiv](https://arxiv.org/pdf/2202.14019.pdf)
  - Off‑the‑shelf pose estimators (OpenPose, SPIN, etc.) are used as baselines, but their outputs are not part of the dataset. [arxiv](https://arxiv.org/pdf/2202.14019.pdf)
- **Download**  
  - Publicly downloadable via the project GitHub / Papers with Code page. [selectdataset](https://www.selectdataset.com/dataset/03e21b9a5b8aa30c20e5435c0b2436ef)
- **Suitability**  
  - Great for **AQA / error detection**, but:
    - **No 2D or 3D joint coordinates are released.**  
    - You could still run MediaPipe/MoveNet/YOLO and assess **self‑consistency or cross‑model variance**, but you wouldn’t have any GT keypoints to compare against.  
  - So Fitness‑AQA does **not meet your 2D ground‑truth requirement.**

### NTU RGB+D (and NTU RGB+D 120)

- **Format / context**  
  - Very large Kinect V2–based action dataset: 56,880 samples (60 classes) in NTU RGB+D; 114,480 samples (120 classes) in NTU RGB+D 120. [emergentmind](https://www.emergentmind.com/topics/ntu-rgb-d-120-dataset)
  - Actions mostly **daily activities, interactions, and some medical signs** (e.g., back pain, falling), but **not structured rehab exercises**. [pdfs.semanticscholar](https://pdfs.semanticscholar.org/b1ee/90f1aa9b380f5400b233dee2e4f0a33e71c7.pdf)
  - For each sample you get **RGB video (1920×1080), depth maps, IR, and 3D skeletal data (25 joints per frame)**. [pdfs.semanticscholar](https://pdfs.semanticscholar.org/b1ee/90f1aa9b380f5400b233dee2e4f0a33e71c7.pdf)
  - Some documentation notes that the skeleton stream includes **joint 3D coordinates plus corresponding pixel indices in RGB & depth planes**, effectively giving 2D image‑plane joint locations. [emergentmind](https://www.emergentmind.com/topics/ntu-rgb-d-120-dataset)
- **Joints & annotations**  
  - Kinect 3D skeleton: 25 joints per frame (spine, neck, shoulders, elbows, wrists, hips, knees, ankles, etc.). [emergentmind](https://www.emergentmind.com/topics/ntu-rgb-d-120-dataset)
  - 2D: not a separate COCO‑style annotation file, but **2D pixel indices can be extracted from the skeleton metadata**, so a 2D skeleton is derivable for all frames. [emergentmind](https://www.emergentmind.com/topics/ntu-rgb-d-120-dataset)
- **Download**  
  - You must **register, submit a request, and accept a release agreement**, after which the lab validates and grants access. [pdfs.semanticscholar](https://pdfs.semanticscholar.org/b1ee/90f1aa9b380f5400b233dee2e4f0a33e71c7.pdf)
- **Suitability**  
  - Technically:
    - You can run MediaPipe/MoveNet/YOLO on RGB and compare joints to the **Kinect 2D/3D skeleton**, so it *can* support 2D evaluation.  
  - However, it **fails two of your framing constraints**:
    - Not specifically rehab / physio exercises (mostly general actions, plus a few “back pain” gestures rather than therapy).  
    - Requires a **formal access request and approval**, which you wanted to avoid. [pdfs.semanticscholar](https://pdfs.semanticscholar.org/b1ee/90f1aa9b380f5400b233dee2e4f0a33e71c7.pdf)
  - If you relax the rehab framing and access constraints, you could still add a brief **“general‑action robustness”** subsection with NTU, but it’s not a clean choice for your main cross‑dataset rehab analysis.

### Physio2.2M (Rode et al., 2025)

- **Format / context**  
  - Dataset of **2.2 million RGB frames** from 25 unimpaired participants performing various physical exercises, captured with 4 RGB cameras plus a 27‑camera Vicon system. [bohrium](https://www.bohrium.com/scholar/8o21908V/Robert_Riener)
  - 3D joint centers (hip, knees, ankles, shoulders, elbows, wrists) reconstructed via Plug‑in Gait and synchronized with RGB at 30 Hz. [github](https://github.com/petteriTeikari/KiMoRe_wrapper)
- **Joints & annotations**  
  - Internally, the study uses these 3D joints to compute both 2D and 3D pose errors for multiple HPE models, but in the **Data Availability** statement they write:
    - “The dataset analyzed during the current study is **not publicly available** due to privacy protection” and is **only available from the corresponding author on reasonable request.** [github](https://github.com/petteriTeikari/KiMoRe_wrapper)
- **Download**  
  - No public URL for videos or joint data; distribution only upon request to the authors. [github](https://github.com/petteriTeikari/KiMoRe_wrapper)
- **Suitability**  
  - Excellent from a **design** perspective (multi‑view RGB + Vicon + 2D projections), but it **fails your “publicly downloadable” hard constraint**.  
  - Unless your supervisors are okay with a “we attempted but could not obtain access in time” note, it’s not usable for an April 2026 bachelor thesis.

***

## Other “almost” options worth knowing about

Two additional datasets you didn’t list but might see in the literature:

- **PhysioNet multi‑camera gait/posture with smart walker**  
  - Depth cameras on a rehab walker + Xsens IMU skeleton; authors provide **aligned 3D and projected 2D joints** in CSVs, but only **depth, not RGB**. [data.mendeley](https://data.mendeley.com/datasets/ygpdzx52g2/1)
  - Great for smart‑walker gait analysis; **not suitable for your RGB‑based mobile HPE comparison.**

- **FLEX (fitness AQA, 2025)**  
  - Multimodal fitness dataset with **5‑view RGB, 3D pose, sEMG, physiological data, 20 weight‑training actions, 38 subjects**. [arxiv](https://arxiv.org/html/2506.03198v1)
  - Geared to AQA; access requires signing a license agreement (CC BY‑NC 4.0) and at time of writing primarily exposes 3D pose and AQA labels. [haoyin116.github](https://haoyin116.github.io/FLEX_Dataset/)
  - Promising but relatively heavy for a bachelor thesis, and again not a clean rehab framing.

***

## Practical recommendation for your thesis

Given all of this, a pragmatic plan that respects your constraints and timeline would be:

1. **Use KERAAL** as your main **cross‑dataset generalization dataset**:
   - Evaluate MediaPipe/MoveNet/YOLOv8‑Pose against **OpenPose COCO 2D** on full‑body joints, and optionally against **projected Vicon/Kinect 3D** for the subset where calibration is feasible. [arxiv](https://arxiv.org/html/2407.00521v2)
   - Emphasize that the 2D reference is “strong baseline” rather than perfect ground truth, and discuss bias from using one HPE model as reference for others.

2. **Optionally add UCO Physical Rehabilitation** as a **limb‑level MoCap‑based benchmark**:
   - Focus on knee and shoulder joints (hip–knee–ankle, shoulder–elbow–wrist) and compare models’ 2D errors in different poses (seated, supine, standing) and camera views. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648737/)
   - This gives you a second dataset where 2D is **true MoCap projection**, at the cost of only having 3 joints per limb and needing an email request.

3. Clearly document in your “Methods – Datasets” chapter:
   - Why **IntelliRehabDS, UI‑PRMD, KIMORE, Fitness‑AQA, NTU RGB+D, and Physio2.2M** were considered but rejected (no RGB, no 2D, access restrictions, non‑rehab context), with one sentence and citation each. [zenodo](https://zenodo.org/records/4610859)

If you’d like, I can help you next with a **short “Datasets” subsection draft** (LaTeX or Markdown) that describes REHAB24‑6 + KERAAL (+ optionally UCO) in the style of a bachelor thesis, including a joint‑mapping table for COCO → your 12‑joint subset.