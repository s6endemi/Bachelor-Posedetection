# Pose Estimation Accuracy Comparison

Bachelorarbeit: **"Einfluss des Körper-Rotationswinkels auf die Genauigkeit von 2D Pose Estimation"**

Systematischer Vergleich von MediaPipe, MoveNet und YOLOv8-Pose zur Analyse der rotationsabhängigen Genauigkeit.

## Setup

```bash
# Virtual Environment erstellen
python -m venv .venv

# Aktivieren (Windows)
.venv\Scripts\activate

# Aktivieren (Linux/Mac)
source .venv/bin/activate

# Dependencies installieren
pip install -r requirements.txt
```

## Projektstruktur

```
bachelor/
├── src/pose_evaluation/      # Hauptcode
│   ├── estimators/           # Pose Estimation Modelle
│   ├── evaluation/           # Metriken und Pipeline
│   └── utils/                # Hilfsfunktionen
├── data/                     # Dataset (nicht in Git)
│   ├── videos/
│   ├── gt_2d/
│   └── gt_3d/
├── results/                  # Ergebnisse
├── notebooks/                # Jupyter Notebooks
└── bachelor.md               # Thesis-Planung
```

## Verwendung

```python
from pose_evaluation.estimators import MediaPipeEstimator, MoveNetEstimator, YOLOPoseEstimator
from pose_evaluation.evaluation import EvaluationPipeline

# Estimatoren initialisieren
estimators = [
    MediaPipeEstimator(model_complexity=2),
    MoveNetEstimator(model_name="thunder"),
    YOLOPoseEstimator(model_size="m"),
]

# Pipeline ausführen
pipeline = EvaluationPipeline(estimators)
results = pipeline.run_single_video(video_path, gt_2d, rotation_angles)
```

## Dataset

REHAB24-6 von Zenodo: https://zenodo.org/records/13305826

## Lizenz

MIT
