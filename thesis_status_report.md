# Thesis Status Report

**Pose Estimation for Rehabilitation Applications**

*January 2026*

---

## Executive Summary

Evaluation of three pose estimation models (MediaPipe, MoveNet, YOLOv8-Pose) on the REHAB24-6 clinical dataset (126 videos, 120,000 frames).

**Key Results:**
- MoveNet: Best accuracy (12.7% error)
- MediaPipe: Most robust in multi-person scenarios (2x better than others)
- Rotation analysis: Limited by dataset structure (bimodal, not continuous)

---

## 1. Project Setup

**Goal:** Find the best pose estimation model for smartphone-based physiotherapy guidance.

**Dataset:** REHAB24-6 (Cernek et al., 2025)

| Property | Value |
| --- | --- |
| Videos | 126 |
| Cameras | 2 (frontal + lateral) |
| Frames evaluated | ~120,000 |
| Ground Truth | Motion Capture |

**Models Tested:**

| Model | Developer | Selection Strategy |
| --- | --- | --- |
| MediaPipe Pose | Google | Torso size |
| MoveNet MultiPose | Google/TensorFlow | Bounding box area |
| YOLOv8-Pose Nano | Ultralytics | Bounding box area |

**Why 2D (not 3D)?**
- MoveNet and YOLO only output 2D
- MediaPipe's 3D has known depth estimation issues
- REHAB24-6 paper states: "Depth estimation is the main limitation"

---

## 2. Key Findings

### 2.1 Model Accuracy (clean data, n=121 videos)

| Model | NMPJPE (Error) | Interpretation |
| --- | --- | --- |
| MoveNet | 12.7% | Best overall |
| MediaPipe | 14.4% | Good, stable |
| YOLO | 17.0% | Third place |

*NMPJPE = error as % of torso length. 10% means ~5cm deviation per joint.*

### 2.2 Selection Robustness (when coach enters frame)

| Model | Normal | With Coach | Error Increase |
| --- | --- | --- | --- |
| MediaPipe | 14.4% | 45.4% | +215% |
| MoveNet | 12.7% | 62.2% | +390% |
| YOLO | 17.0% | 66.0% | +289% |

**Insight:** MediaPipe's torso-based selection is ~2x more robust than bounding-box selection. Torso size correlates better with camera distance than bounding box area (which measures "spread", not "size").

### 2.3 Camera Perspective Effect

| Model | Frontal (c17) | Lateral (c18) | Difference |
| --- | --- | --- | --- |
| MediaPipe | 15.2% | 13.6% | +1.6% |
| MoveNet | 14.1% | 11.4% | +2.8% |
| YOLO | 20.4% | 14.0% | +6.5% |

Effect is statistically significant (p < 0.001) but practically small (Cohen's d < 0.2).

---

## 3. Dataset Limitation: The Rotation Problem

**Original Hypothesis:** Analyze how body rotation (0° frontal to 90° lateral) affects accuracy.

**Reality:** Dataset has almost no continuous rotation.

| Rotation Range | % of Frames | Description |
| --- | --- | --- |
| 0-30° (frontal) | 41.2% | Mostly camera c17 |
| 30-60° (diagonal) | 17.6% | Very sparse |
| 60-90° (lateral) | 41.3% | Mostly camera c18 |

**Problem:** Patients don't rotate during exercises - they either face the camera OR stand sideways. The data is bimodal, not continuous. This makes "rotation effect analysis" essentially just "camera 1 vs camera 2 comparison."

---

## 4. Technical Challenges Solved

During implementation, several problems were discovered and fixed:

| Problem | Cause | Solution |
| --- | --- | --- |
| MediaPipe 29% detection failures | Default confidence=0.5 too strict | Lowered to 0.1 |
| Models tracking wrong person | No multi-person handling | Implemented selection strategies |
| MoveNet couldn't handle multi-person | SinglePose architecture limitation | Switched to MultiPose |
| Rotation angles didn't match visuals | MoCap coordinates ≠ camera coordinates | Empirical transformation (offset=65°) |
| Evaluation showed wrong numbers | Frame alignment bug (frame_step=3 not applied) | Rewrote evaluation script |

**Key Insight:** Each problem initially looked like a model weakness but turned out to be an implementation issue. This is important for reproducibility.

---

## 5. What Exists Now

**Code & Data:**
- Inference pipeline for all 3 models
- 126 prediction files (.npz)
- Frame-level evaluation data (120k rows)
- Statistical analysis (ANOVA, t-tests, effect sizes)

**Documentation:**
- Methodology documentation
- Problem/solution log
- Results with corrected numbers

**Visualizations:**
- 7 publication-ready figures (PNG + PDF)

---

## 6. Open Questions

### Possible Thesis Directions

| Option | Focus | Pro | Con |
| --- | --- | --- | --- |
| A | Practical model comparison | Directly applicable | Might seem incremental |
| B | Selection strategy analysis | Novel finding | Small sample (n=5 coach videos) |
| C | Dataset limitations study | Methodologically interesting | Might seem negative |
| D | Extension (more models/data) | More comprehensive | More work needed |

### Questions for Supervisor

1. Is the selection robustness finding significant enough?
2. How to frame the "rotation hypothesis didn't work" situation?
3. Extend the work or focus on existing results?
4. Related work to compare against?

---

## 7. Quick Reference

```
Dataset:         126 videos, 120k frames, 12 comparable joints
Best Accuracy:   MoveNet (12.7%)
Most Robust:     MediaPipe (2x better at multi-person)
Camera Effect:   +1.6% to +6.5% (small practical significance)
Rotation Data:   Bimodal (41% frontal, 18% middle, 41% lateral)
Coach Videos:    5 of 126 (4%)
```

---

*End of Report*
