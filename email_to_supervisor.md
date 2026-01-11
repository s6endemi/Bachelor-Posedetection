# Email an Betreuer

**Subject:** Bachelor Thesis - Status Update & Need for Direction

---

Dear [Supervisor Name],

I wanted to update you on my thesis and ask for your guidance on how to proceed from here.

## What I Did

I evaluated three pose estimation models (MediaPipe, MoveNet, YOLOv8-Pose) on the REHAB24-6 dataset - 126 clinical physiotherapy videos with motion capture ground truth, roughly 120,000 frames evaluated.

## The Dataset

REHAB24-6 is in principle ideal for my use case: real patients doing actual physiotherapy exercises, recorded from two camera angles, with high-quality motion capture as ground truth. It's one of the few clinical rehabilitation datasets publicly available.

However, during implementation I ran into several limitations:
- **Rotation:** Patients mostly stand either frontal or sideways - almost no continuous rotation during exercises
- **Camera calibration:** No camera position data provided, so I had to empirically determine the coordinate transformation
- **Multi-person:** In some videos, a physiotherapist walks into frame, causing the models to sometimes track the wrong person

These aren't flaws in the dataset itself - it was designed for exercise recognition, not specifically for the kind of pose estimation analysis I wanted to do.

## Results

Basic model comparison worked:
- MoveNet: 12.7% error (best)
- MediaPipe: 14.4% error
- YOLO: 17.0% error

The rotation analysis I originally planned didn't work - the data is too bimodal (41% frontal, 41% lateral, only 18% in between).

Camera perspective: After deeper analysis, frontal (c17) is actually slightly better than lateral for typical frames. However, c17 has 5-18x more "person-switch" frames where the model briefly tracks the wrong person. After filtering these extreme outliers, frontal is 1-2% better - as expected.

This revealed something important: The 5 videos I identified with a coach in frame were only the worst cases. There are ~26 additional videos with sporadic multi-person frames that cause occasional tracking errors. This multi-person robustness issue appears to be more pervasive than initially thought.

Selection strategy finding: When a second person enters the frame, models with bounding-box selection (MoveNet, YOLO) had +290-390% error increase, while MediaPipe with torso-based selection only had +215%.

## Where I Am Now

The technical work is done - implementation, evaluation, statistical analysis, visualizations. What I'm missing is a clear direction for framing this as a thesis.

Possible angles:
- Practical model comparison for telehealth
- Selection robustness analysis (if n=5 is enough)
- Dataset limitations as methodological insight
- Some combination
- Or extend the work if needed

I'm not sure which of these has enough substance for a bachelor thesis.

## Questions

- What do you see as the most viable path forward?
- Should I extend the work in some direction?
- Any related work I should look at?

I've attached a detailed report with methodology and complete results.

Best regards,
[Your Name]

---

**Attachment:** `thesis_status_report.pdf`
