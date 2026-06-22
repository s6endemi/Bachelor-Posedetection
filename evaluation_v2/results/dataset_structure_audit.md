# Dataset Structure Audit

This note records a critical mismatch between the current thesis wording and the
dataset structure visible in the repository.

## In-Repo Structure

- `evaluation_v2/results/rehab24/frame_results.csv` contains `63` unique
  `(video_id, exercise)` pairs.
- Each of those pairs appears with exactly `2` cameras (`c17`, `c18`),
  producing `126` evaluated view combinations.
- `data/Segmentation (1).csv` maps all `63` `(video_id, exercise)` pairs to
  `10` unique `person_id` clusters.

## External Dataset Description

The official Masaryk University dataset description for
`REHAB24-6: A multi-modal dataset of physical rehabilitation exercises`
describes the dataset as:

- `65 recordings`
- `6 common rehabilitation exercises`
- `10 subjects`

## Consequence

The current thesis wording that describes the dataset as
`21 patients x 6 exercises x 2 cameras = 126 videos` is incompatible with both:

- the repository's own result structure, and
- the official dataset description above.

## Working Rule For The Recovery

Until the thesis wording is corrected, analysis scripts should treat the
segmentation `person_id` as the best available in-repo clustering variable for
subject-level aggregation.

This does not yet resolve every wording issue in Chapter 3, but it prevents us
from building new statistical claims on top of a clearly inconsistent dataset
count.
