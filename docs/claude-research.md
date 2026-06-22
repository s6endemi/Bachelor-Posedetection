# Systematic Literature Search: Mobile Pose Estimation for Home-Based Rehabilitation

This systematic literature search identifies **25 peer-reviewed papers** (2017-2025) to support a bachelor thesis evaluating MediaPipe BlazePose, MoveNet Lightning, and YOLOv8s-Pose on the REHAB24-6 dataset. Papers are organized by the eight identified literature gaps, with full citations, DOIs, and relevance descriptions.

---

## Gap 1: NMPJPE metric definition and origin

Three papers establish the theoretical foundation for normalized pose error metrics used in clinical evaluation.

### 1.1 Zhang, D., Wu, Y., Guo, M., & Chen, Y. (2021)
**Title:** Deep Learning Methods for 3D Human Pose Estimation under Different Supervision Paradigms: A Survey  
**Venue:** *Electronics*, 10(18), 2267  
**DOI:** 10.3390/electronics10182267

**Relevance:** This comprehensive survey explicitly defines NMPJPE as "MPJPE calculated after aligning the depths of the root joints" in Human3.6M Protocol 1. The paper clarifies distinctions between NMPJPE, PA-MPJPE (Procrustes-aligned), and P-MPJPE (reconstruction error), providing foundational metric definitions essential for standardized rehabilitation pose evaluation.

### 1.2 Rode, D., Dunkel, A., Willi, R., Wolf, P., Xiloyannis, M., & Riener, R. (2025)
**Title:** Assessment of monocular human pose estimation models for clinical movement analysis  
**Venue:** *Scientific Reports*, 15, 38767  
**DOI:** 10.1038/s41598-025-22626-7

**Relevance:** Directly relevant to home-based rehabilitation, this study evaluates 11 open-source monocular HPE models for clinical movement analysis. Reports MPJPE ranges of **72-122mm** in 2D and **146-249mm** in 3D. The Physio2.2M dataset with physical exercise movements addresses the need for rehabilitation-specific evaluation metrics and standardized clinical assessment approaches.

### 1.3 Nakano, N., Sakura, T., Ueda, K., Omura, L., Kimura, A., Iino, Y., Fukashiro, S., & Yoshioka, S. (2020)
**Title:** Evaluation of 3D Markerless Motion Capture Accuracy Using OpenPose With Multiple Video Cameras  
**Venue:** *Frontiers in Sports and Active Living*, 2, 50  
**DOI:** 10.3389/fspor.2020.00050

**Relevance:** Evaluates markerless 3D motion capture accuracy using Mean Absolute Error metrics for rehabilitation and sports applications. Discusses applicability of pose estimation metrics for clinical biomechanics, noting that "deep-learning-based markerless motion capture is expected to be applied to sporting games and rehabilitations."

---

## Gap 2: Rotation and viewpoint robustness

Three papers quantify how camera viewing angles affect pose estimation accuracy—critical for home rehabilitation where camera placement varies.

### 2.1 Jahn, L., Flügge, S., Zhang, D., Poustka, L., Bölte, S., Wörgötter, F., Marschik, P.B., & Kulvicius, T. (2025)
**Title:** Comparison of marker-less 2D image-based methods for infant pose estimation  
**Venue:** *Scientific Reports*, 15, Article 12148  
**DOI:** 10.1038/s41598-025-96206-0

**Relevance:** Directly quantifies viewing angle effects comparing top-down versus diagonal camera positions using ViTPose, OpenPose, MediaPipe, and HRNet. Found significantly better accuracy from top-down views, with **hip keypoints showing the most pronounced accuracy differences** between viewing angles. Provides empirical evidence for camera placement recommendations in clinical settings.

### 2.2 Ajoje, O.O., Mbada, C.E., Akinwande, O.O., Fatoye, F., & Olagbegi, O.M. (2021)
**Title:** Concurrent validity of human pose tracking in video for measuring gait parameters in older adults: a preliminary analysis with multiple trackers, viewing angles, and walking directions  
**Venue:** *Journal of NeuroEngineering and Rehabilitation*, 18, 163  
**DOI:** 10.1186/s12984-021-00933-0

**Relevance:** Validation study comparing AlphaPose, OpenPose, and Detectron against 3D motion capture for gait analysis in older adults. Explicitly examines multiple camera viewing angles (front versus back at various positions) and their effects on clinical gait monitoring accuracy, demonstrating that correlation with gold-standard measurements varies significantly by viewing angle.

### 2.3 Mehta, D., Rhodin, H., Casas, D., Fua, P., Sotnychenko, O., Xu, W., & Theobalt, C. (2017)
**Title:** Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision  
**Venue:** *International Conference on 3D Vision (3DV)*, pp. 506-516. IEEE  
**DOI:** 10.1109/3DV.2017.00064

**Relevance:** Introduces the MPI-INF-3DHP dataset with **14 cameras at different angles**, specifically designed to address viewpoint-invariance limitations. The paper notes that models trained on chest-height camera datasets struggle with complex viewing angles, making this dataset valuable for evaluating pose estimation robustness across viewpoints.

---

## Gap 3: Multi-person robustness and person selection

Three papers address accuracy degradation when multiple persons (e.g., patient and physiotherapist) appear in the frame.

### 3.1 Li, J., Wang, C., Zhu, H., Mao, Y., Fang, H.S., & Lu, C. (2019)
**Title:** CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark  
**Venue:** *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 10863-10872  
**DOI:** 10.1109/CVPR.2019.01112

**Relevance:** Demonstrates that state-of-the-art methods show approximately **20 mAP accuracy drop** in crowded versus uncrowded scenes. Introduces the "Crowd Index" metric to quantify crowding level and proposes solutions for joint misattribution between nearby persons. The CrowdPose benchmark (20,000 images, 80,000 persons) enables systematic evaluation directly relevant to rehabilitation scenarios where therapists may be present.

### 3.2 Liu, Q., Zhang, Y., Bai, S., & Yuille, A. (2022)
**Title:** Explicit Occlusion Reasoning for Multi-person 3D Human Pose Estimation  
**Venue:** *European Conference on Computer Vision (ECCV)*, LNCS vol. 13665, pp. 497-517  
**DOI:** 10.1007/978-3-031-20065-6_29

**Relevance:** Addresses occlusion in multi-person scenarios by explicitly modeling how to infer occluded joints from visible cues. Identifies four common failure modes: extra person detection, missing person, incomplete skeleton, and wrong position estimates. Achieves **6.0 PCK improvement** over bottom-up methods and **2.8 PCK** over top-down methods—highly relevant for rehabilitation where one person may partially occlude another.

### 3.3 Chen, H., Guo, P., Li, P., Lee, G.H., & Chirikjian, G. (2020)
**Title:** Multi-person 3D Pose Estimation in Crowded Scenes Based on Multi-View Geometry  
**Venue:** *European Conference on Computer Vision (ECCV)*, LNCS vol. 12346, pp. 541-557  
**DOI:** 10.1007/978-3-030-58452-8_32

**Relevance:** Addresses person identification challenges where poses become associated with keypoints from different persons, causing inaccurate person counts. Multi-view approach helps disambiguate joint-to-person assignment, relevant for home rehabilitation requiring correct target person identification.

---

## Gap 4: Temporal stability and jitter metrics

Two papers define the mathematical framework for evaluating frame-to-frame keypoint consistency, essential for smooth motion tracking.

### 4.1 Zeng, A., Yang, L., Ju, X., Li, J., Wang, J., & Xu, Q. (2022)
**Title:** SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos  
**Venue:** *European Conference on Computer Vision (ECCV)*, pp. 625-642  
**DOI:** 10.1007/978-3-031-20065-6_36  
**arXiv:** 2112.13715

**Relevance:** **Primary reference for temporal jitter metrics.** Introduces **Acceleration Error (Accel)** as the key metric for temporal smoothness, defined as the difference between predicted and ground-truth acceleration vectors in mm/s². Addresses "highly-unbalanced jitters" in video pose estimation and proposes a temporal refinement network—critical for rehabilitation applications where smooth motion tracking ensures accurate movement assessment.

### 4.2 Zhang, Y., Li, Z., An, L., Li, M., Yu, T., & Liu, Y. (2024)
**Title:** Hybrid 3D Human Pose Estimation with Monocular Video and Sparse IMUs  
**Venue:** *Computer Vision and Image Understanding*  
**arXiv:** 2404.17837

**Relevance:** Provides explicit definitions for temporal stability metrics: **MPJAE (Mean Per Joint Acceleration Error)** evaluates second-order differential accuracy of predicted 3D poses, while **MPJJE (Mean Per Joint Jitter Error)** is calculated from the first derivative of joint acceleration, directly demonstrating motion smoothness. These metrics directly apply to rehabilitation movement analysis quality assessment.

---

## Gap 5: Top-down versus bottom-up versus one-stage paradigms

Three comprehensive surveys systematically define and compare HPE architecture families, essential for contextualizing the thesis model selection.

### 5.1 Zheng, C., Wu, W., Chen, C., Yang, T., Zhu, S., Shen, J., Kehtarnavaz, N., & Shah, M. (2023)
**Title:** Deep Learning-Based Human Pose Estimation: A Survey  
**Venue:** *ACM Computing Surveys*, 56(1), 1-37  
**DOI:** 10.1145/3603618

**Relevance:** Most comprehensive HPE survey covering **250+ papers**. Explicitly defines top-down methods (detect person first via Faster R-CNN, then estimate pose within bounding box) versus bottom-up methods (detect all keypoints first, then group using Part Affinity Fields or associative embeddings). Key finding: "top-down pipeline yields better results... bottom-up methods are generally faster."

### 5.2 Gao, Z., Chen, J., Liu, Y., Jin, Y., & Tian, D. (2025)
**Title:** A systematic survey on human pose estimation: upstream and downstream tasks, approaches, lightweight models, and prospects  
**Venue:** *Artificial Intelligence Review*, 58(3), 68  
**DOI:** 10.1007/s10462-024-11060-2

**Relevance:** Recent comprehensive survey covering both 2D and 3D HPE methods through 2023-2024. Explicitly categorizes multi-person methods into top-down and bottom-up approaches while discussing single-stage regression methods. Particularly valuable for including **lightweight model considerations** relevant to mobile deployment.

### 5.3 Neupane, D., Bhattarai, A., Aryal, S., & Seok, J. (2024)
**Title:** A survey on deep 3D human pose estimation  
**Venue:** *Artificial Intelligence Review*  
**DOI:** 10.1007/s10462-024-11019-3

**Relevance:** Explicitly states: "The problem-solving paradigms used in 3D multi-person pose estimation can be divided into **top-down, bottom-up, and hybrid approaches**." Covers CNNs, GCNs, and Transformers with detailed accuracy/speed comparisons. Discusses monocular setups and multi-person scenarios relevant to rehabilitation applications.

---

## Gap 6: Confidence thresholding

Three papers address optimal confidence threshold selection, directly supporting the thesis's use of 0.5 threshold.

### 6.1 Gu, K., Yang, L., Yao, A., & Van Gool, L. (2023)
**Title:** On the Calibration of Human Pose Estimation  
**Venue:** *arXiv preprint*  
**arXiv:** 2311.17105

**Relevance:** **First paper to systematically address miscalibration in pose estimation.** Demonstrates that "most 2D human pose estimation frameworks estimate keypoint confidence in an ad-hoc manner, using heuristics such as the maximum value of heatmaps." Shows that confidence should align with pose accuracy (OKS) and proposes Calibrated ConfidenceNet (CCNet) achieving **up to 1.4% AP improvement** through proper calibration.

### 6.2 Ienaga, N., Takahata, S., Terayama, K., Enomoto, D., Ishihara, H., Noda, H., & Hagihara, H. (2022)
**Title:** Development and Verification of Postural Control Assessment Using Deep-Learning-Based Pose Estimators: Towards Clinical Applications  
**Venue:** *Occupational Therapy International*, 2022, 6952999  
**DOI:** 10.1155/2022/6952999

**Relevance:** **Directly validates the 0.5 threshold for rehabilitation.** Compares confidence thresholds across OpenPose, AlphaPose, and MediaPipe Pose. Key finding: "keypoints with visibility less than 0.5 were considered outliers." For MediaPipe Pose specifically, "error distance decreased slowly, especially when threshold was set above 0.5"—providing empirical justification for the thesis's threshold choice.

### 6.3 Samkari, E., Arif, M., Alghamdi, M., & Al Ghamdi, M.A. (2023)
**Title:** Human Pose Estimation Using Deep Learning: A Systematic Literature Review  
**Venue:** *Machine Learning and Knowledge Extraction*, 5(4), 1612-1659  
**DOI:** 10.3390/make5040081

**Relevance:** Systematic review (2014-2023) covering 100+ HPE papers. Discusses evaluation metrics including Object Keypoint Similarity (OKS) that underlies confidence threshold selection. Explains that OKS thresholds of **0.5 (loose) and 0.75 (strict)** are standard benchmarks: "All papers use the threshold value of 0.5 or 0.75," establishing field conventions.

---

## Gap 7: Lightweight and mobile HPE comparison

Four papers specifically compare mobile-deployable pose estimation models for practical clinical applications.

### 7.1 Jo, B. & Kim, S. (2022)
**Title:** Comparative Analysis of OpenPose, PoseNet, and MoveNet Models for Pose Estimation in Mobile Devices  
**Venue:** *Traitement du Signal*, 39(1), 119-124  
**DOI:** 10.18280/ts.390111

**Relevance:** Directly compares four pose estimation models (OpenPose, PoseNet, MoveNet Lightning, MoveNet Thunder) specifically for mobile device deployment. Key findings: **MoveNet Lightning was fastest**, OpenPose slowest, PoseNet achieved 97.6% accuracy. Essential for establishing which lightweight models perform best on resource-constrained devices.

### 7.2 Chung, J.L., Ong, L.Y., & Leow, M.C. (2022)
**Title:** Comparative Analysis of Skeleton-Based Human Pose Estimation  
**Venue:** *Future Internet*, 14(12), 380  
**DOI:** 10.3390/fi14120380

**Relevance:** Comprehensive comparison of four state-of-the-art HPE libraries (OpenPose, PoseNet, MoveNet, MediaPipe Pose) using image and video datasets. **MoveNet showed best performance** for both static images and videos. Explicitly discusses clinical applications including in-home rehabilitation, medical assistance, and physiotherapy exercise evaluation.

### 7.3 Hii, C.S.T., Gan, K.B., Zainal, N., Mohamed Ibrahim, N., Azmin, S., Mat Desa, S.H., van de Warrenburg, B., & You, H.W. (2023)
**Title:** Automated Gait Analysis Based on a Marker-Free Pose Estimation Model  
**Venue:** *Sensors*, 23(14), 6489  
**DOI:** 10.3390/s23146489

**Relevance:** Validates MediaPipe Pose against Vicon optoelectronic gold-standard for gait analysis, demonstrating **good to excellent agreement (ICC >0.75)** for spatiotemporal parameters. Highlights MediaPipe's top-down approach for precise landmark detection. Directly applicable to home-based rehabilitation requiring marker-free, accessible pose estimation.

### 7.4 Stenum, J., Rossi, C., & Roemmich, R.T. (2021)
**Title:** Two-dimensional video-based analysis of human gait using pose estimation  
**Venue:** *PLoS Computational Biology*, 17(4), e1008935  
**DOI:** 10.1371/journal.pcbi.1008935

**Relevance:** Compares OpenPose with 3D motion capture for gait analysis, showing small errors in temporal parameters and step lengths. Provides an accessible, accurate workflow suitable for broader gait analysis applications beyond specialized labs, demonstrating feasibility of deploying pose estimation in non-laboratory home settings.

---

## Gap 8: Home-based rehabilitation motivation

Four papers provide statistics and evidence supporting the need for technology-assisted remote physiotherapy monitoring.

### 8.1 Jack, K., McLean, S.M., Moffett, J.K., & Gardiner, E. (2010)
**Title:** Barriers to treatment adherence in physiotherapy outpatient clinics: A systematic review  
**Venue:** *Manual Therapy*, 15(3), 220-228  
**DOI:** 10.1016/j.math.2009.10.004

**Relevance:** Foundational systematic review identifying barriers to physiotherapy adherence across 20 high-quality studies. Strong evidence that **non-adherence reaches 50%** and is associated with low physical activity, low self-efficacy, depression, anxiety, and poor social support. Establishes the critical need for technology-enhanced solutions to improve home exercise adherence.

### 8.2 Zhang, Z.Y., Tian, L., He, K., et al. (2022)
**Title:** Digital Rehabilitation Programs Improve Therapeutic Exercise Adherence for Patients With Musculoskeletal Conditions: A Systematic Review With Meta-Analysis  
**Venue:** *Journal of Orthopaedic & Sports Physical Therapy*, 52(11), 726-739  
**DOI:** 10.2519/jospt.2022.11384

**Relevance:** Meta-analysis demonstrating that **digital rehabilitation programs significantly improve adherence** to therapeutic exercise for musculoskeletal conditions. Shows digital health technologies facilitate healthcare access and improve individual adherence in home-based settings—directly supporting the thesis motivation.

### 8.3 Simmich, J., Ross, M.H., & Russell, T. (2024)
**Title:** Real-time video telerehabilitation shows comparable satisfaction and similar or better attendance and adherence compared with in-person physiotherapy: a systematic review  
**Venue:** *Journal of Physiotherapy*, 70(3), 181-192  
**DOI:** 10.1016/j.jphys.2024.05.001

**Relevance:** Systematic review of RCTs (n=1,247 participants) showing telerehabilitation resulted in **8% higher attendance** and **9% higher adherence** to exercise programs compared to in-person physiotherapy, with similar satisfaction. Provides strong evidence that video-based telerehabilitation is a viable alternative for physiotherapy delivery.

### 8.4 Saaei, F. & Klappa, S.G. (2021)
**Title:** Rethinking Telerehabilitation: Attitudes of Physical Therapists and Patients  
**Venue:** *Journal of Physical Therapy Education*, 35(3), 231-242  
**DOI:** 10.1177/23743735211034335

**Relevance:** Survey of 228 physical therapists showing **73% reported increased telerehabilitation use** during COVID-19. Identifies that AI-powered motion tracking offers cost-effective solutions for tracking patient adherence with real-time feedback. Discusses how telerehabilitation increases patient engagement through home convenience and reduces travel barriers.

---

## Summary table

| Gap | Topic | Papers Found | Key References |
|-----|-------|-------------|----------------|
| 1 | NMPJPE metric definition | 3 | Zhang et al. 2021, Rode et al. 2025 |
| 2 | Viewpoint robustness | 3 | Jahn et al. 2025, Ajoje et al. 2021 |
| 3 | Multi-person robustness | 3 | Li et al. 2019 (CrowdPose), Liu et al. 2022 |
| 4 | Temporal jitter metrics | 2 | Zeng et al. 2022 (SmoothNet) |
| 5 | HPE paradigms comparison | 3 | Zheng et al. 2023 (ACM Surveys) |
| 6 | Confidence thresholding | 3 | Ienaga et al. 2022, Gu et al. 2023 |
| 7 | Lightweight HPE comparison | 4 | Jo & Kim 2022, Chung et al. 2022 |
| 8 | Home rehabilitation motivation | 4 | Jack et al. 2010, Simmich et al. 2024 |

**Total: 25 peer-reviewed papers** (verified against exclusion list)

---

## Recommendations for thesis integration

**Methodology section:** Cite Zeng et al. 2022 for acceleration jitter metric, Ienaga et al. 2022 for 0.5 confidence threshold justification, and Zhang et al. 2021 for NMPJPE definition.

**Related work section:** Use Zheng et al. 2023 and Gao et al. 2025 for HPE paradigm taxonomy; cite Li et al. 2019 for multi-person accuracy degradation quantification.

**Introduction/motivation:** Jack et al. 2010 provides the foundational 50% non-adherence statistic; Simmich et al. 2024 offers recent evidence that telerehabilitation improves outcomes.

**Model comparison:** Jo & Kim 2022 and Chung et al. 2022 provide direct precedent for comparing MoveNet, MediaPipe, and similar lightweight models in practical applications.