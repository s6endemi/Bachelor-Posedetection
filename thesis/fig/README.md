# thesis/fig - figure assets

## Current Thesis Figures

The following figures are current and referenced from the thesis text:

- `fig_accuracy_boxplot.pdf` / `.png`
- `fig_cross_dataset_slopegraph.pdf` / `.png`
- `fig_rotation_line.pdf` / `.png`
- `coach_detection_comparison.png` / `.jpg`
- `fig_prediction_displacement_boxplot.pdf` / `.png`
- `fig_speed_accuracy_scatter.pdf` / `.png`
- `failure_mode_taxonomy.pdf`

The temporal-stability figure uses normalized frame-to-frame prediction
displacement. This metric is a stability proxy for the predicted trajectories; it
does not subtract ground-truth motion.

## Archived Figures

Older figures from previous thesis iterations are kept in `archive_old/`. They
are not part of the current thesis build and should not be used as evidence for
the final results.

## Canonical Data

Figure regeneration should use the current canonical outputs under
`evaluation_v2/results/`, especially:

- `evaluation_v2/results/rehab24/frame_results.csv`
- `evaluation_v2/results/rehab24_all/frame_results_all.csv`
- `evaluation_v2/results/rehab24/patient_level_stats.json`
- `evaluation_v2/results/rehab24/patient_level_pairwise_stats.csv`
- `evaluation_v2/results/rehab24/jitter_results.csv`
- `evaluation_v2/results/rehab24_all/jitter_results_all.csv`
- `evaluation_v2/results/coach_detection_counts_summary.csv`
- `evaluation_v2/results/hip_sensitivity_summary.csv`
- `evaluation_v2/results/coco/coco_results.csv`
- `evaluation_v2/results/inference_benchmark_all.csv`
