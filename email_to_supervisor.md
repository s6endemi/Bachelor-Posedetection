# Email to Supervisor

**Subject:** Bachelor Thesis – Technical Work Complete, Seeking Your Input

---

Dear [Supervisor Name],

The technical work for my thesis is complete. I wanted to share my results and get your feedback before I start writing.

**What I've done:**
- Built a full evaluation pipeline for three mobile pose estimation models (MediaPipe, MoveNet, YOLOv8-Pose)
- Analyzed 363,529 frames from 126 clinical rehabilitation videos (REHAB24-6 dataset, MoCap ground truth)
- Completed statistical analysis, visualizations, and a systematic literature comparison (5 relevant papers)

**About the dataset:**
REHAB24-6 is ideal in principle – real rehabilitation patients, physiotherapy exercises, two camera angles, high-quality ground truth. In practice, it revealed limitations for my original focus: the rotation distribution is bimodal (patients stand frontal OR sideways, rarely in between), and multi-person scenarios are limited. This isn't a flaw – the dataset was designed for exercise recognition, not pose estimation analysis.

**What this means for the thesis:**
A narrow focus on rotation effects alone doesn't provide enough depth. However, the data supports a broader evaluation: comparing mobile HPE models across accuracy, temporal stability, and robustness for home-based rehabilitation. This framing aligns well with the practical needs of mobile physio apps (like Previa Health).

**Key findings (details in attached report):**
- MediaPipe ≈ MoveNet in accuracy (not statistically distinguishable)
- MediaPipe is more robust to rotation (+31% vs +54% degradation) and multi-person scenarios
- MoveNet is more stable frame-to-frame (42% less jitter)
- YOLO has the highest detection rate (88% complete keypoints vs 64% for MediaPipe)
- No single model dominates – trade-offs depend on the use case

**My proposal:**
Frame the thesis as a comprehensive evaluation for home-based rehabilitation, with practical recommendations for mobile apps. I believe this is a stronger contribution than forcing a narrow angle the data doesn't fully support.

I've attached a detailed report with all results and methodology. Happy to send additional materials (visualizations, raw data, code) if helpful.

Let me know what you think – happy to adjust the scope or focus if needed. Would be great to get your go-ahead before I start writing.

Best regards,
[Name]

---

**Attachment:** `thesis_status_report.pdf`
