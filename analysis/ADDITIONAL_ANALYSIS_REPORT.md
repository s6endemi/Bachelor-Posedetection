# Additional Analysis Report: Temporal Stability & Detection Quality

> **Date:** 13. Januar 2026
> **Analysis:** Temporal Jitter + Valid-Joints Correlation
> **Data:** 363,529 frames (clean, without coach videos, NMPJPE < 100%)

---

## 1. Temporal Jitter Analysis

### 1.1 What is Jitter?

**Temporal Jitter** measures frame-to-frame stability of pose predictions. Lower jitter = smoother, more stable predictions = better for real-time feedback in apps.

```
Jitter = |NMPJPE(frame_t) - NMPJPE(frame_t-1)|
```

### 1.2 Overall Results

| Model | Mean Jitter | Median Jitter | Std | P90 | P95 | N Transitions |
|-------|-------------|---------------|-----|-----|-----|---------------|
| **MoveNet** | **1.06%** | **0.42%** | 3.12% | 1.99% | 3.50% | 117,043 |
| YOLO | 1.12% | 0.38% | 3.20% | 2.12% | 3.91% | 113,301 |
| MediaPipe | 1.51% | 0.53% | 4.53% | 2.48% | 5.19% | 117,091 |

**Key Finding:** MoveNet is the most stable model (lowest jitter), MediaPipe is the least stable.

### 1.3 Statistical Significance

| Comparison | Mean Diff | t-stat | p-value | Significant |
|------------|-----------|--------|---------|-------------|
| MediaPipe vs MoveNet | +0.449% | 27.93 | <0.0001 | **Yes** |
| MediaPipe vs YOLO | +0.390% | 23.83 | <0.0001 | **Yes** |
| MoveNet vs YOLO | -0.059% | -4.47 | <0.0001 | **Yes** |

All differences are statistically significant. MediaPipe has ~42% higher jitter than MoveNet.

### 1.4 Jitter by Camera

| Camera | MediaPipe | MoveNet | YOLO |
|--------|-----------|---------|------|
| c17 (frontal) | 1.65% | **0.88%** | **0.74%** |
| c18 (lateral) | 1.39% | 1.24% | 1.45% |

**Key Finding:** All models are more stable on c17 (frontal) than c18 (lateral). The difference is most pronounced for MoveNet and YOLO.

### 1.5 Jitter by Rotation

| Rotation | MediaPipe | MoveNet | YOLO | Interpretation |
|----------|-----------|---------|------|----------------|
| 0-10° | 1.11% | **0.51%** | 0.52% | Most stable |
| 10-20° | 2.33% | 0.68% | 0.66% | MediaPipe unstable |
| 20-30° | 0.67% | 0.68% | 0.71% | All stable |
| 30-40° | 0.66% | 0.64% | 0.70% | All stable |
| 50-60° | 1.51% | 1.46% | 1.24% | Increasing |
| 60-70° | 2.07% | 1.35% | 1.23% | Higher |
| 70-80° | 2.24% | 1.52% | 2.11% | Higher |
| 80-90° | 1.72% | 1.88% | 2.03% | Highest for MoveNet/YOLO |

**Key Finding:**
- Frontal angles (0-40°) have lower jitter
- Lateral angles (70-90°) have higher jitter
- MediaPipe shows unusual spike at 10-20° (possibly due to c17 Multi-Person issues)

### 1.6 Deep Dive: Why Does MediaPipe Have Higher Jitter?

We investigated the root cause of MediaPipe's higher jitter through correlation analysis.

#### Hypothesis 1: Detection Instability Causes Jitter

When joints "flip" (appear/disappear between frames), the error calculation changes, causing jitter.

| Model | Frames with Joint Changes | Jitter when stable | Jitter when joints flip | Ratio |
|-------|---------------------------|-------------------|------------------------|-------|
| MediaPipe | 12.2% | 1.19% | 3.85% | **3.2x** |
| MoveNet | 13.8% | 0.69% | 3.42% | 5.0x |
| YOLO | **3.3%** | 1.00% | 4.74% | 4.7x |

**Finding:** Joint flips cause 3-5x higher jitter. But MediaPipe's "stable" jitter (1.19%) is still higher than MoveNet's (0.69%), suggesting an architectural difference beyond detection instability.

#### Hypothesis 2: Which Joints Are Unstable?

We counted how often each joint's detection status changes between consecutive frames.

| Joint | MediaPipe Flip Rate | MoveNet Flip Rate | YOLO Flip Rate |
|-------|---------------------|-------------------|----------------|
| **Right Knee** | **3.44%** | 0.32% | 0.00% |
| **Right Ankle** | **2.81%** | 0.90% | 0.00% |
| Left Wrist | 2.37% | **5.07%** | 0.02% |
| Right Wrist | 2.14% | **5.14%** | 2.06% |
| Shoulders/Hips | ~0% | ~0% | ~0% |

**Finding:**
- MediaPipe struggles with **right-side lower body** (knee, ankle)
- MoveNet struggles with **wrists** (both sides)
- YOLO is most stable across all joints

#### Hypothesis 3: Rotation Increases Instability

| Rotation | MP Joint Changes | MN Joint Changes |
|----------|------------------|------------------|
| 0-10° (frontal) | 1.9% | 1.5% |
| 30-40° | 4.2% | 6.0% |
| **80-90° (lateral)** | **24.9%** | **33.9%** |

**Finding:** At lateral view (80-90°), one quarter to one third of frames have joint detection changes. This explains why jitter increases at extreme angles.

#### Multi-Person Effect on Stability

| Model | Clean Videos | Coach Videos | Change |
|-------|--------------|--------------|--------|
| MediaPipe | 12.2% | **30.8%** | **+150%** |
| MoveNet | 13.8% | 22.1% | +60% |
| YOLO | 3.3% | 0.5% | -85% |

**Interpretation:**
- MediaPipe's better selection (torso-based) causes more person-switching → more instability
- YOLO's worse selection causes it to "stick" to one person → more stable but often wrong person
- This reveals a **trade-off: correct person vs. stable prediction**

### 1.7 Practical Implications

For real-time physiotherapy feedback:
- **MoveNet** provides most stable predictions (important for tracking exercises over time)
- **YOLO** is similarly stable but may track wrong person in multi-person scenarios
- **MediaPipe** has more frame-to-frame variation (apply smoothing filter: moving average over 3-5 frames)
- **All models** become unstable at lateral angles (>70°) → guide users to frontal positioning

---

## 2. Valid Joints vs Error Analysis

### 2.1 What are "Valid Joints"?

Each frame has 12 potential joints. If a model cannot confidently detect a joint, it's marked as invalid. Fewer valid joints = lower confidence = potentially higher error.

### 2.2 Correlation Results

| Model | Pearson r | p-value | Interpretation |
|-------|-----------|---------|----------------|
| MediaPipe | **-0.220** | <0.001 | Moderate negative correlation |
| MoveNet | -0.185 | <0.001 | Moderate negative correlation |
| YOLO | -0.066 | <0.001 | Weak negative correlation |

**Interpretation:** For all models, fewer valid joints correlates with higher error. MediaPipe shows the strongest relationship.

### 2.3 Error by Number of Valid Joints

#### MediaPipe
| Valid Joints | N Frames | Mean NMPJPE | Median NMPJPE |
|--------------|----------|-------------|---------------|
| 12 | 74,978 | 11.2% | 10.6% |
| 11 | 5,473 | 15.3% | 13.0% |
| 10 | 20,414 | 15.4% | 12.5% |
| 8-9 | 16,263 | 14.1% | 13.1% |
| <8 | 84 | ~55% | ~50% |

#### MoveNet
| Valid Joints | N Frames | Mean NMPJPE | Median NMPJPE |
|--------------|----------|-------------|---------------|
| 12 | 92,769 | 10.9% | 10.1% |
| 11 | 13,682 | 14.0% | 12.4% |
| 10 | 7,036 | 13.5% | 11.8% |
| <10 | 3,677 | ~15% | ~12% |

#### YOLO
| Valid Joints | N Frames | Mean NMPJPE | Median NMPJPE |
|--------------|----------|-------------|---------------|
| 12 | 99,638 | 12.7% | 11.1% |
| 11 | 3,229 | 17.8% | 15.1% |
| 10 | 10,552 | 13.6% | 11.8% |

### 2.4 Summary: Can Valid Joints Predict Quality?

| Model | Frames with <12 Joints | Mean Error (<12) | Mean Error (12) | Error Increase |
|-------|------------------------|------------------|-----------------|----------------|
| MediaPipe | 42,234 (36.0%) | 15.0% | 11.2% | **+34%** |
| MoveNet | 24,395 (20.8%) | 13.8% | 10.9% | +27% |
| YOLO | 13,784 (12.2%) | 14.6% | 12.7% | +15% |

**Key Findings:**
1. **MediaPipe** has the highest rate of incomplete detections (36% of frames)
2. **YOLO** has the lowest rate (12%)
3. Filtering frames with <12 joints would:
   - MediaPipe: Remove 36% of frames, improve mean error by 3.8%
   - MoveNet: Remove 21% of frames, improve mean error by 2.8%
   - YOLO: Remove 12% of frames, improve mean error by 1.9%

### 2.5 Practical Recommendation

**Quality Filter for Apps:**
- If `valid_joints < 12`, flag frame as low-confidence
- For MediaPipe: Consider stricter threshold (≤10 joints = very unreliable)
- Trade-off: More filtering = fewer frames but higher quality

---

## 3. Joint Detection Rate Analysis

### 3.1 Which Joints are Most Often Missing?

| Joint | MediaPipe | MoveNet | YOLO | Problem Joint? |
|-------|-----------|---------|------|----------------|
| Shoulders (L/R) | 100.0% | 99.7-99.8% | 99.4-99.6% | No |
| Hips (L/R) | 100.0% | 99.6-99.7% | 99.6-99.7% | No |
| Left Elbow | 90.3% | 95.1% | 99.5% | MediaPipe: Yes |
| **Right Elbow** | **75.3%** | 92.6% | 87.8% | **All: Yes** |
| Left Wrist | 92.0% | 89.6% | 99.4% | MoveNet: Moderate |
| **Right Wrist** | **77.0%** | 89.4% | 90.3% | **MediaPipe: Yes** |
| Left Knee | 98.2% | 99.1% | 98.9% | No |
| **Right Knee** | **86.3%** | 99.3% | 99.0% | **MediaPipe: Yes** |
| Left Ankle | 98.9% | 98.1% | 98.8% | No |
| Right Ankle | 92.4% | 98.3% | 98.8% | MediaPipe: Moderate |

### 3.2 Key Findings

1. **MediaPipe struggles with right-side joints:**
   - Right Elbow: 75.3% (vs 92-99% for others)
   - Right Wrist: 77.0% (vs 89-90% for others)
   - Right Knee: 86.3% (vs 99% for others)

2. **All models struggle with Right Elbow** (likely occlusion issue)

3. **YOLO has most consistent detection** across all joints (87-99%)

### 3.3 Why Right-Side Problems?

Possible explanations:
- **Camera positioning:** c17/c18 may favor left-side visibility
- **Exercise movements:** Patients may turn with right side away from camera
- **Occlusion:** Right arm/leg may be occluded by body during exercises

---

## 4. Combined Insights for Previa Health

### 4.1 Model Recommendations (Corrected)

| Criterion | Best Model | Margin | Statistically Significant? |
|-----------|------------|--------|---------------------------|
| **Accuracy** | MoveNet | 0.8% better | **No** (p=0.098) |
| **Stability (Jitter)** | MoveNet | 42% less | Yes |
| **Rotation Robustness** | **MediaPipe** | **~50% more stable** | Yes |
| **Multi-Person Robustness** | **MediaPipe** | **~40% better** | Yes |
| **Detection Rate** | YOLO | 24% more complete | Yes |
| **Keypoints** | MediaPipe | 33 vs 17 | N/A |

**Key Insight:** MediaPipe and MoveNet are statistically equivalent in accuracy. MediaPipe wins on the dimensions that matter most for real-world home-based rehabilitation (rotation, multi-person).

### 4.2 Practical Implementation Guidelines

1. **Frame Quality Filter:**
   ```
   if valid_joints < 12:
       mark_as_low_confidence()
   if valid_joints < 10:
       discard_frame()
   ```

2. **Temporal Smoothing:**
   - MediaPipe benefits most from smoothing (highest jitter)
   - Apply moving average over 3-5 frames for exercise tracking

3. **Right-Side Joint Warning:**
   - If tracking right-side exercises, warn user to position better
   - Or use YOLO for right-side dominant movements

4. **Lateral View Warning:**
   - Jitter increases significantly at >70° rotation
   - App should guide user to frontal position

---

## 5. Files Created

| File | Content |
|------|---------|
| `temporal_jitter.csv` | Jitter statistics per model |
| `valid_joints_analysis.csv` | Error by valid joints count |
| `joint_detection_rate.csv` | Detection rate per joint per model |

---

## 6. Summary Table

| Metric | MediaPipe | MoveNet | YOLO | Winner |
|--------|-----------|---------|------|--------|
| Median NMPJPE | 11.2% | 10.4% | 11.3% | ≈ (not significant) |
| Mean Jitter | 1.51% | **1.06%** | 1.12% | MoveNet |
| Rotation Robustness | **+31%** | +54% | +58% | **MediaPipe** |
| Multi-Person Robustness | **+209%** | +340% | +274% | **MediaPipe** |
| Full Detection Rate | 64.0% | 79.2% | **87.8%** | YOLO |
| Keypoints | **33** | 17 | 17 | **MediaPipe** |

**Overall Recommendation:**
- **Realistic Home Environment (Previa Health):** MediaPipe - more robust to rotation and multi-person
- **Controlled Lab Environment:** MoveNet - slightly better accuracy and stability
- **High Detection Rate Critical:** YOLO - most reliable joint detection

**Key Finding:** No single model dominates. MediaPipe and MoveNet are statistically equivalent in accuracy (p=0.098), but MediaPipe is significantly more robust under suboptimal conditions.
