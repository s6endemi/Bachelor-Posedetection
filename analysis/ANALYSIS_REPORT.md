# Comprehensive Pose Estimation Analysis Report

**Generated:** 2026-01-11 14:10
**Total Frames Analyzed:** 363,529
**Clean Frames:** 351,778
**Coach Frames:** 11,751

---

## 1. Overall Statistics (Clean Data)

| Model | N Frames | Mean | Median | Std | IQR | P90 | P95 |
|-------|----------|------|--------|-----|-----|-----|-----|
| MediaPipe | 118,305 | 14.5% | 11.2% | 23.1% | 3.8% | 17.0% | 22.8% |
| MoveNet | 118,472 | 14.8% | 10.4% | 32.6% | 4.3% | 16.8% | 20.9% |
| YOLO | 115,001 | 17.7% | 11.3% | 42.5% | 4.8% | 20.2% | 27.4% |

## 2. Camera Comparison (Clean Data)

### Mean NMPJPE

| Model | c17 Mean | c17 Median | c18 Mean | c18 Median | Diff (Mean) |
|-------|----------|------------|----------|------------|-------------|
| MediaPipe | 15.5% | 10.8% | 13.6% | 11.7% | +1.8% |
| MoveNet | 16.8% | 9.7% | 12.8% | 11.2% | +4.0% |
| YOLO | 21.3% | 10.5% | 14.5% | 12.1% | +6.8% |

### Outlier Frames (>30%, >50%, >100%)

| Model | Camera | >30% | >50% | >100% | >100% Rate |
|-------|--------|------|------|-------|------------|
| MediaPipe | c17 | 1974 | 1239 | 906 | 1.59% |
| MediaPipe | c18 | 1759 | 855 | 187 | 0.31% |
| MoveNet | c17 | 1614 | 1309 | 1213 | 2.12% |
| MoveNet | c18 | 1126 | 375 | 95 | 0.16% |
| YOLO | c17 | 2049 | 1584 | 1500 | 2.79% |
| YOLO | c18 | 2537 | 720 | 79 | 0.13% |

## 3. Rotation Analysis (10-degree buckets, Clean Data)

### Sample Sizes per Bucket

| Bucket | MediaPipe | MoveNet | YOLO |
|--------|------|------|------|
| 0-10 | 15,839 | 15,914 | 15,334 |
| 10-20 | 13,791 | 13,832 | 13,407 |
| 20-30 | 17,065 | 17,071 | 16,914 |
| 30-40 | 9,487 | 9,488 | 9,460 |
| 40-50 | 2,219 | 2,219 | 2,124 |
| 50-60 | 9,507 | 9,507 | 8,154 |
| 60-70 | 17,834 | 17,862 | 17,025 |
| 70-80 | 16,398 | 16,412 | 16,417 |
| 80-90 | 16,165 | 16,167 | 16,166 |

### Mean NMPJPE per Bucket

| Bucket | MediaPipe | MoveNet | YOLO |
|--------|------|------|------|
| 0-10 | 14.5% | 17.4% | 25.3% |
| 10-20 | 17.5% | 17.9% | 20.9% |
| 20-30 | 11.3% | 11.5% | 12.8% |
| 30-40 | 10.5% | 11.2% | 12.7% |
| 40-50 | 16.8% | 28.7% | 32.5% |
| 50-60 | 13.0% | 13.9% | 16.8% |
| 60-70 | 16.0% | 13.9% | 16.1% |
| 70-80 | 16.4% | 14.1% | 16.5% |
| 80-90 | 14.5% | 15.1% | 17.0% |

### Median NMPJPE per Bucket

| Bucket | MediaPipe | MoveNet | YOLO |
|--------|------|------|------|
| 0-10 | 10.1% | 9.4% | 9.6% |
| 10-20 | 10.4% | 8.5% | 9.3% |
| 20-30 | 10.6% | 10.1% | 10.8% |
| 30-40 | 10.6% | 9.9% | 10.8% |
| 40-50 | 11.1% | 11.2% | 14.2% |
| 50-60 | 10.8% | 10.1% | 11.1% |
| 60-70 | 12.5% | 10.9% | 12.1% |
| 70-80 | 12.4% | 12.2% | 13.1% |
| 80-90 | 13.3% | 13.8% | 14.9% |

## 4. Statistical Significance

**ANOVA:** F=316.30, p=5.69e-138, Significant: Yes

### Pairwise Comparisons (Bonferroni-corrected)

| Comparison | Mean Diff | Cohen's d | p-value | Significant |
|------------|-----------|-----------|---------|-------------|
| MediaPipe vs MoveNet | -0.25% | -0.009 | 9.76e-02 | No |
| MediaPipe vs YOLO | -3.15% | -0.092 | 5.31e-110 | Yes |
| MoveNet vs YOLO | -2.90% | -0.077 | 2.05e-76 | Yes |

## 5. Coach Impact Analysis

| Model | Clean Mean | Coach Mean | Increase | Increase % |
|-------|------------|------------|----------|------------|
| MediaPipe | 14.5% | 44.8% | +30.3% | +209% |
| MoveNet | 14.8% | 64.9% | +50.2% | +340% |
| YOLO | 17.7% | 66.1% | +48.5% | +274% |

## 6. Per-Joint Analysis (Clean Data)

### MediaPipe

| Joint | Mean | Median | Std |
|-------|------|--------|-----|
| left_hip | 19.2% | 17.2% | 25.2% |
| right_hip | 17.7% | 15.8% | 22.7% |
| right_ankle | 16.7% | 11.6% | 29.6% |
| left_ankle | 16.3% | 11.7% | 27.5% |
| left_wrist | 14.1% | 9.5% | 23.6% |
| right_wrist | 13.1% | 9.9% | 21.2% |
| right_elbow | 12.8% | 9.5% | 20.4% |
| right_knee | 12.4% | 8.0% | 23.8% |
| left_elbow | 12.1% | 8.8% | 18.9% |
| right_shoulder | 10.7% | 7.6% | 20.3% |
| left_knee | 10.5% | 6.9% | 21.5% |
| left_shoulder | 10.0% | 7.2% | 24.8% |

### MoveNet

| Joint | Mean | Median | Std |
|-------|------|--------|-----|
| right_wrist | 17.5% | 11.2% | 28.3% |
| left_wrist | 15.7% | 11.6% | 26.3% |
| right_elbow | 14.6% | 9.6% | 24.2% |
| left_elbow | 14.5% | 10.0% | 29.1% |
| left_hip | 14.2% | 11.2% | 30.7% |
| right_shoulder | 13.7% | 10.0% | 28.7% |
| right_hip | 13.1% | 9.9% | 28.6% |
| left_ankle | 13.0% | 10.8% | 18.7% |
| left_shoulder | 12.3% | 8.4% | 35.6% |
| right_ankle | 11.9% | 9.8% | 17.1% |
| left_knee | 9.9% | 6.4% | 25.8% |
| right_knee | 9.6% | 6.2% | 23.8% |

### YOLO

| Joint | Mean | Median | Std |
|-------|------|--------|-----|
| right_wrist | 21.1% | 12.6% | 35.1% |
| left_wrist | 19.4% | 11.9% | 40.0% |
| left_hip | 19.0% | 14.7% | 41.2% |
| right_hip | 17.2% | 13.9% | 31.9% |
| right_elbow | 16.2% | 10.5% | 27.6% |
| left_elbow | 16.0% | 9.5% | 40.3% |
| left_ankle | 13.3% | 10.1% | 20.7% |
| right_shoulder | 13.3% | 9.7% | 27.8% |
| left_shoulder | 12.9% | 8.0% | 41.2% |
| right_ankle | 12.9% | 9.5% | 20.7% |
| right_knee | 11.6% | 8.0% | 20.6% |
| left_knee | 10.0% | 6.6% | 20.3% |

## 7. Top Outlier Videos (>30% NMPJPE frames)

| Video | Model | Camera | Coach? | Outlier Frames | Rate | Max NMPJPE |
|-------|-------|--------|--------|----------------|------|------------|
| PM_121-c17 | YOLO | c17 | Yes | 456 | 81.3% | 576.6% |
| PM_010-c17 | MediaPipe | c17 | Yes | 334 | 24.6% | 461.4% |
| PM_112-c18 | YOLO | c18 | No | 314 | 21.2% | 85.2% |
| PM_010-c17 | YOLO | c17 | Yes | 305 | 22.5% | 390.5% |
| PM_010-c17 | MoveNet | c17 | Yes | 301 | 22.2% | 401.4% |
| PM_112-c18 | MediaPipe | c18 | No | 294 | 19.8% | 167.2% |
| PM_108-c17 | MoveNet | c17 | Yes | 259 | 27.5% | 525.2% |
| PM_042-c18 | YOLO | c18 | No | 257 | 21.3% | 65.9% |
| PM_121-c17 | MoveNet | c17 | Yes | 250 | 43.9% | 578.9% |
| PM_121-c17 | MediaPipe | c17 | Yes | 240 | 42.2% | 38.2% |
| PM_025-c17 | MediaPipe | c17 | No | 236 | 27.3% | 489.2% |
| PM_028-c18 | MediaPipe | c18 | No | 212 | 17.5% | 73.3% |
| PM_021-c18 | YOLO | c18 | No | 209 | 22.0% | 65.0% |
| PM_027-c17 | YOLO | c17 | No | 197 | 14.4% | 341.9% |
| PM_112-c17 | MoveNet | c17 | No | 197 | 13.3% | 487.5% |
| PM_021-c17 | MediaPipe | c17 | No | 191 | 20.1% | 382.4% |
| PM_042-c18 | MediaPipe | c18 | No | 191 | 15.9% | 64.6% |
| PM_025-c18 | YOLO | c18 | No | 190 | 21.9% | 125.2% |
| PM_027-c18 | YOLO | c18 | No | 186 | 13.6% | 133.8% |
| PM_003-c17 | MediaPipe | c17 | No | 176 | 11.1% | 177.6% |
