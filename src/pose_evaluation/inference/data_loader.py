"""
Data Loader for REHAB24-6 Dataset.

Findet alle Video-GT Paare und laedt sie.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import re


@dataclass
class VideoSample:
    """Ein Video mit zugehoerigen Ground Truth Daten."""
    video_path: Path
    gt_2d_path: Path
    gt_3d_path: Path
    exercise: str      # Ex1, Ex2, ...
    subject_id: str    # PM_000, PM_001, ...
    camera: str        # c17 oder c18

    def load_gt_2d(self) -> np.ndarray:
        """Laedt 2D Ground Truth. Shape: (num_frames, 26, 2)"""
        return np.load(self.gt_2d_path)

    def load_gt_3d(self) -> np.ndarray:
        """Laedt 3D Ground Truth. Shape: (num_frames, 26, 4)"""
        return np.load(self.gt_3d_path)


class DataLoader:
    """Laedt alle Video-GT Paare aus dem REHAB24-6 Dataset."""

    def __init__(self, data_root: Path):
        """
        Args:
            data_root: Pfad zum data/ Ordner
        """
        self.data_root = Path(data_root)
        self.videos_dir = self.data_root / "videos"
        self.gt_2d_dir = self.data_root / "gt_2d"
        self.gt_3d_dir = self.data_root / "gt_3d"

    def discover_samples(self) -> list[VideoSample]:
        """
        Findet alle gueltigen Video-GT Paare.

        Returns:
            Liste von VideoSample Objekten
        """
        samples = []

        # Alle Exercise-Ordner durchgehen
        for exercise_dir in sorted(self.videos_dir.iterdir()):
            if not exercise_dir.is_dir():
                continue

            exercise = exercise_dir.name  # Ex1, Ex2, ...

            # Alle Videos im Exercise-Ordner
            for video_path in sorted(exercise_dir.glob("*.mp4")):
                sample = self._parse_video(video_path, exercise)
                if sample:
                    samples.append(sample)

        return samples

    def _parse_video(self, video_path: Path, exercise: str) -> VideoSample | None:
        """
        Parst ein Video und findet die zugehoerigen GT-Dateien.

        Naming Convention:
        - Video: PM_XXX-Camera17-30fps.mp4 oder PM_XXX-Camera18-30fps-transposed.mp4
        - GT 2D: PM_XXX-c17-30fps.npy oder PM_XXX-c18-30fps.npy
        - GT 3D: PM_XXX-30fps.npy
        """
        filename = video_path.name

        # Pattern: PM_XXX-Camera{17,18}-30fps...
        match = re.match(r"(PM_\d+)-Camera(\d+)-30fps", filename)
        if not match:
            return None

        subject_id = match.group(1)  # PM_000
        camera_num = match.group(2)  # 17 oder 18
        camera = f"c{camera_num}"    # c17 oder c18

        # GT Pfade konstruieren
        gt_2d_path = self.gt_2d_dir / exercise / f"{subject_id}-{camera}-30fps.npy"
        gt_3d_path = self.gt_3d_dir / exercise / f"{subject_id}-30fps.npy"

        # Pruefen ob GT existiert
        if not gt_2d_path.exists():
            print(f"[WARN] GT 2D nicht gefunden: {gt_2d_path}")
            return None

        if not gt_3d_path.exists():
            print(f"[WARN] GT 3D nicht gefunden: {gt_3d_path}")
            return None

        return VideoSample(
            video_path=video_path,
            gt_2d_path=gt_2d_path,
            gt_3d_path=gt_3d_path,
            exercise=exercise,
            subject_id=subject_id,
            camera=camera
        )

    def get_samples_by_exercise(self, exercise: str) -> list[VideoSample]:
        """Filtert Samples nach Exercise."""
        return [s for s in self.discover_samples() if s.exercise == exercise]

    def get_samples_by_camera(self, camera: str) -> list[VideoSample]:
        """Filtert Samples nach Kamera (c17 oder c18)."""
        return [s for s in self.discover_samples() if s.camera == camera]


if __name__ == "__main__":
    # Test
    loader = DataLoader(Path("data"))
    samples = loader.discover_samples()

    print(f"Gefunden: {len(samples)} Video-GT Paare")
    print()

    # Nach Exercise gruppieren
    from collections import Counter
    exercises = Counter(s.exercise for s in samples)
    cameras = Counter(s.camera for s in samples)

    print("Nach Exercise:")
    for ex, count in sorted(exercises.items()):
        print(f"  {ex}: {count} Videos")

    print("\nNach Kamera:")
    for cam, count in sorted(cameras.items()):
        print(f"  {cam}: {count} Videos")

    # Beispiel
    if samples:
        s = samples[0]
        print(f"\nBeispiel: {s.video_path.name}")
        print(f"  GT 2D: {s.gt_2d_path.name}")
        print(f"  GT 3D: {s.gt_3d_path.name}")

        gt_2d = s.load_gt_2d()
        gt_3d = s.load_gt_3d()
        print(f"  2D Shape: {gt_2d.shape}")
        print(f"  3D Shape: {gt_3d.shape}")
