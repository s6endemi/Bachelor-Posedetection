# Canonical Provenance Report

This file freezes the current thesis evidence base during the recovery process.

## Authoritative Files

- `evaluation_v2/results/rehab24/frame_results.csv`
- `evaluation_v2/results/rehab24_all/frame_results_all.csv`
- `evaluation_v2/results/coco/coco_results.csv`

## Subset Definitions

- `clean`: No detection failure.
- `valid_rehab`: No detection failure, no >100% outlier, no MoCap annotation error.
- `valid_coco`: No detection failure, no >100% outlier.
- `coach_extreme_case`: 5 identified coach videos on c17: PM_010, PM_011, PM_108, PM_119, PM_121.
- `segmentation_multi_person`: Rows with extra_person_severity >= 2.

## Rehab24 Main Models (Valid Subset)

| Model | Mean NMPJPE | Median | Std | Frames |
|-------|-------------|--------|-----|--------|
| MoveNet | 10.52% | 9.85% | 4.66% | 119439 |
| MediaPipe | 12.53% | 11.26% | 7.05% | 119729 |
| YOLOv8 | 12.77% | 11.23% | 6.73% | 115711 |

## Rehab24 All Variants (Valid Subset)

| Model | Mean NMPJPE | Median | Frames |
|-------|-------------|--------|--------|
| MoveNet_MultiPose | 10.52% | 9.85% | 119439 |
| YOLOv8_Medium | 10.86% | 10.01% | 114591 |
| YOLOv8_Small | 11.43% | 10.27% | 116009 |
| MediaPipe_Heavy | 11.51% | 10.59% | 119630 |
| MediaPipe_Full | 12.53% | 11.26% | 119729 |
| YOLOv8_Nano | 12.77% | 11.23% | 115711 |
| MoveNet_SP_Thunder | 13.33% | 11.93% | 117719 |
| MediaPipe_Lite | 13.54% | 11.77% | 119392 |
| MoveNet_SP_Lightning | 14.97% | 12.83% | 94395 |

## COCO (Valid Subset)

| Model | Mean NMPJPE | Median | Frames |
|-------|-------------|--------|--------|
| YOLOv8m | 10.15% | 7.63% | 1326 |
| YOLOv8s | 10.88% | 8.10% | 1319 |
| MoveNet_SP_Thunder | 12.01% | 8.00% | 1087 |
| MoveNet | 12.44% | 8.71% | 1213 |
| MoveNet_SP_Lightning | 12.63% | 8.49% | 821 |
| YOLOv8 | 13.69% | 10.06% | 1318 |
| MediaPipe_Heavy | 15.08% | 8.72% | 1133 |
| MediaPipe | 17.21% | 10.52% | 1101 |
| MediaPipe_Lite | 20.07% | 12.83% | 1108 |

## Warnings

- Segmentation metadata maps 63 video/exercise pairs to 10 person_id clusters; thesis dataset-size claims should be reconciled before final writing.
