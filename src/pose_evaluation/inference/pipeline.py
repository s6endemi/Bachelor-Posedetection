"""
Inference Pipeline - Fuehrt Pose Estimation auf allen Videos aus.
"""

import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Generator
import json
from datetime import datetime

from .data_loader import DataLoader, VideoSample
from ..estimators import PoseEstimator


@dataclass
class FrameResult:
    """Ergebnis fuer einen einzelnen Frame."""
    frame_idx: int
    predictions: dict[str, np.ndarray]  # model_name -> (17, 3) keypoints
    gt_2d: np.ndarray                   # (12, 2) vergleichbare keypoints
    rotation_angle: float               # Winkel in Grad


@dataclass
class VideoResult:
    """Ergebnis fuer ein ganzes Video."""
    sample: VideoSample
    num_frames: int
    predictions: dict[str, np.ndarray]  # model_name -> (num_frames, 17, 3)
    rotation_angles: np.ndarray         # (num_frames,) in Grad


class InferencePipeline:
    """
    Haupt-Pipeline fuer die Pose Estimation Evaluation.

    Fuehrt alle Modelle auf allen Videos aus und speichert die Ergebnisse.
    """

    # GT Indices fuer Schultern (fuer Rotationsberechnung)
    LEFT_SHOULDER_IDX = 7   # LeftArm
    RIGHT_SHOULDER_IDX = 12  # RightArm

    # Kamera-Offset: Bei diesem MoCap-Winkel steht Person frontal zu Camera 17
    # Empirisch bestimmt aus PM_114, PM_122, PM_109 (siehe docs/02_PROBLEMS_AND_SOLUTIONS.md)
    C17_FRONTAL_OFFSET = 65.0

    def __init__(
        self,
        estimators: list[PoseEstimator],
        data_root: Path,
        output_dir: Path
    ):
        """
        Args:
            estimators: Liste der Pose Estimators
            data_root: Pfad zum data/ Ordner
            output_dir: Pfad fuer Ergebnisse
        """
        self.estimators = estimators
        self.data_loader = DataLoader(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_rotation_angle(self, gt_3d_frame: np.ndarray, camera: str) -> float:
        """
        Berechnet den Rotationswinkel RELATIV ZUR KAMERA aus 3D GT.

        Args:
            gt_3d_frame: 3D Keypoints fuer einen Frame, Shape (26, 4)
            camera: Kamera-ID ('c17' oder 'c18')

        Returns:
            Kamera-relativer Rotationswinkel in Grad (0 = frontal, 90 = seitlich)
        """
        left_shoulder = gt_3d_frame[self.LEFT_SHOULDER_IDX]
        right_shoulder = gt_3d_frame[self.RIGHT_SHOULDER_IDX]

        # Schulterachse im Raum
        dx = right_shoulder[0] - left_shoulder[0]
        dz = right_shoulder[2] - left_shoulder[2]

        # MoCap-Winkel berechnen
        mocap_angle = np.degrees(np.arctan2(abs(dz), abs(dx)))

        # Transformation zu kamera-relativem Winkel
        # c17 sieht Person als frontal bei MoCap ~65 Grad
        # c18 ist 90 Grad zu c17 gedreht
        c17_relative = abs(mocap_angle - self.C17_FRONTAL_OFFSET)

        if camera == 'c17':
            return c17_relative
        else:  # c18
            return 90.0 - c17_relative

    def process_frame(
        self,
        frame: np.ndarray,
        gt_3d_frame: np.ndarray,
        camera: str
    ) -> tuple[dict[str, np.ndarray], float]:
        """
        Verarbeitet einen einzelnen Frame.

        Args:
            frame: Video-Frame als numpy array
            gt_3d_frame: 3D Ground Truth fuer diesen Frame
            camera: Kamera-ID ('c17' oder 'c18')

        Returns:
            (predictions dict, rotation_angle)
        """
        predictions = {}

        for estimator in self.estimators:
            keypoints = estimator.predict(frame)

            # In numpy array konvertieren (17, 3) - x, y, confidence
            kp_array = np.array([
                [kp.x, kp.y, kp.confidence]
                for kp in keypoints
            ])
            predictions[estimator.get_model_name()] = kp_array

        rotation_angle = self.calculate_rotation_angle(gt_3d_frame, camera)

        return predictions, rotation_angle

    def process_video(
        self,
        sample: VideoSample,
        max_frames: int | None = None,
        progress_callback: callable = None,
        frame_step: int = 1
    ) -> VideoResult:
        """
        Verarbeitet ein komplettes Video.

        Args:
            sample: VideoSample mit Pfaden
            max_frames: Optional - nur erste N frames verarbeiten
            progress_callback: Optional - callback(current, total)
            frame_step: Nur jeden N-ten Frame verarbeiten (default: 1 = alle)

        Returns:
            VideoResult mit allen Predictions
        """
        # GT laden
        gt_2d = sample.load_gt_2d()
        gt_3d = sample.load_gt_3d()

        # Video oeffnen
        cap = cv2.VideoCapture(str(sample.video_path))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Berechne welche Frames wir verarbeiten
        max_source_frames = min(total_video_frames, len(gt_3d))
        if max_frames:
            max_source_frames = min(max_source_frames, max_frames * frame_step)

        # Frame-Indices die wir verarbeiten (0, step, 2*step, ...)
        frame_indices = list(range(0, max_source_frames, frame_step))
        num_output_frames = len(frame_indices)

        # Ergebnis-Arrays initialisieren
        model_names = [e.get_model_name() for e in self.estimators]
        predictions = {name: np.zeros((num_output_frames, 17, 3)) for name in model_names}
        rotation_angles = np.zeros(num_output_frames)

        # Frames verarbeiten
        output_idx = 0
        video_frame_idx = 0

        while output_idx < num_output_frames:
            target_frame = frame_indices[output_idx]

            # Frames skippen bis zum Ziel-Frame
            while video_frame_idx < target_frame:
                ret, _ = cap.read()
                if not ret:
                    break
                video_frame_idx += 1

            # Ziel-Frame lesen
            ret, frame = cap.read()
            if not ret:
                break
            video_frame_idx += 1

            # Sicherstellen dass GT-Daten vorhanden
            if target_frame >= len(gt_3d):
                break

            # Frame verarbeiten (mit Kamera-Info fuer kamera-relative Winkel)
            frame_preds, rotation = self.process_frame(frame, gt_3d[target_frame], sample.camera)

            # Speichern
            for model_name, kps in frame_preds.items():
                predictions[model_name][output_idx] = kps
            rotation_angles[output_idx] = rotation

            if progress_callback:
                progress_callback(output_idx + 1, num_output_frames)

            output_idx += 1

        cap.release()

        # Arrays auf tatsaechliche Laenge kuerzen
        for name in model_names:
            predictions[name] = predictions[name][:output_idx]
        rotation_angles = rotation_angles[:output_idx]

        return VideoResult(
            sample=sample,
            num_frames=output_idx,
            predictions=predictions,
            rotation_angles=rotation_angles
        )

    def save_result(self, result: VideoResult):
        """Speichert VideoResult als .npz Datei."""
        # Output Pfad: output_dir/Ex1/PM_000-c17.npz
        out_dir = self.output_dir / result.sample.exercise
        out_dir.mkdir(exist_ok=True)

        filename = f"{result.sample.subject_id}-{result.sample.camera}.npz"
        out_path = out_dir / filename

        # Predictions + Metadata speichern
        save_dict = {
            "rotation_angles": result.rotation_angles,
            "num_frames": result.num_frames,
        }

        # Predictions pro Modell
        for model_name, preds in result.predictions.items():
            # Modellname bereinigen fuer Dateiname
            safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
            save_dict[f"pred_{safe_name}"] = preds

        np.savez_compressed(out_path, **save_dict)
        return out_path

    def run(
        self,
        max_videos: int | None = None,
        max_frames_per_video: int | None = None,
        exercises: list[str] | None = None,
        skip_existing: bool = True,
        frame_step: int = 1
    ):
        """
        Fuehrt die komplette Pipeline aus.

        Args:
            max_videos: Optional - nur erste N Videos
            max_frames_per_video: Optional - nur erste N Frames pro Video
            exercises: Optional - nur bestimmte Exercises (z.B. ["Ex1", "Ex2"])
            skip_existing: Ueberspringe bereits verarbeitete Videos (default: True)
            frame_step: Nur jeden N-ten Frame verarbeiten (default: 1 = alle)
        """
        samples = self.data_loader.discover_samples()

        # Filtern
        if exercises:
            samples = [s for s in samples if s.exercise in exercises]

        if max_videos:
            samples = samples[:max_videos]

        # Skip existing
        if skip_existing:
            original_count = len(samples)
            samples_to_process = []
            for s in samples:
                out_path = self.output_dir / s.exercise / f"{s.subject_id}-{s.camera}.npz"
                if not out_path.exists():
                    samples_to_process.append(s)
            samples = samples_to_process
            skipped = original_count - len(samples)
            if skipped > 0:
                print(f"Skipping {skipped} already processed videos")

        print(f"Processing {len(samples)} videos...")
        print(f"Models: {[e.get_model_name() for e in self.estimators]}")
        if frame_step > 1:
            print(f"Frame step: {frame_step} (processing every {frame_step}. frame)")
        print()

        results_summary = []

        for i, sample in enumerate(samples):
            print(f"[{i+1}/{len(samples)}] {sample.video_path.name}")

            def progress(current, total):
                pct = current / total * 100
                print(f"\r  Frame {current}/{total} ({pct:.1f}%)", end="", flush=True)

            result = self.process_video(
                sample,
                max_frames=max_frames_per_video,
                progress_callback=progress,
                frame_step=frame_step
            )
            print()  # Newline nach Progress

            out_path = self.save_result(result)
            print(f"  -> Saved: {out_path}")

            results_summary.append({
                "video": sample.video_path.name,
                "exercise": sample.exercise,
                "camera": sample.camera,
                "frames": result.num_frames,
                "output": str(out_path)
            })

        # Summary speichern
        summary_path = self.output_dir / "inference_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "models": [e.get_model_name() for e in self.estimators],
                "frame_step": frame_step,
                "total_videos": len(samples),
                "results": results_summary
            }, f, indent=2)

        print(f"\nDone! Summary: {summary_path}")


if __name__ == "__main__":
    # Quick test mit einem Video
    from ..estimators import MediaPipeEstimator, MoveNetEstimator, YOLOPoseEstimator

    estimators = [
        MediaPipeEstimator(model_complexity=1),
        MoveNetEstimator(model_variant="thunder"),
        YOLOPoseEstimator(model_size="n"),  # nano fuer schnellen Test
    ]

    pipeline = InferencePipeline(
        estimators=estimators,
        data_root=Path("data"),
        output_dir=Path("data/predictions")
    )

    # Nur 1 Video, 10 Frames zum Testen
    pipeline.run(max_videos=1, max_frames_per_video=10)
