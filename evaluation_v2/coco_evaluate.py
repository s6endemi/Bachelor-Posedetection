"""
COCO val2017 Evaluation — Cross-Dataset Comparison

Runs MediaPipe, MoveNet, YOLOv8 on COCO val2017 images and computes
NMPJPE against COCO ground truth keypoints.

Usage:
    python evaluation_v2/coco_evaluate.py --test          # 20 images, verify pipeline
    python evaluation_v2/coco_evaluate.py                 # full evaluation
    python evaluation_v2/coco_evaluate.py --variants      # include model variants
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_v2.config import COMPARABLE_COCO_INDICES, COMPARABLE_JOINT_NAMES


# =============================================================================
# COCO Data Loading
# =============================================================================

def load_coco_annotations(anno_path: Path) -> tuple[dict, list]:
    """Load COCO keypoint annotations. Returns (image_lookup, annotations)."""
    with open(anno_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    annos_by_image = {}
    for ann in coco["annotations"]:
        annos_by_image.setdefault(ann["image_id"], []).append(ann)

    return images, annos_by_image


def get_gt_keypoints(annotation: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 17 COCO keypoints and visibility from annotation.

    Returns:
        keypoints: (17, 2) array of [x, y]
        visibility: (17,) array where 0=not labeled, 1=not visible, 2=visible
    """
    kps = np.array(annotation["keypoints"]).reshape(17, 3)
    return kps[:, :2], kps[:, 2]


def compute_torso_length_coco(gt_kps: np.ndarray, gt_vis: np.ndarray) -> float:
    """Torso length from COCO keypoints (shoulder-center to hip-center)."""
    # COCO: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
    if gt_vis[5] < 2 or gt_vis[6] < 2 or gt_vis[11] < 2 or gt_vis[12] < 2:
        return np.nan  # not all torso joints visible

    shoulder_center = (gt_kps[5] + gt_kps[6]) / 2
    hip_center = (gt_kps[11] + gt_kps[12]) / 2
    length = np.linalg.norm(shoulder_center - hip_center)
    return length if length > 1.0 else np.nan


# =============================================================================
# Model Loading
# =============================================================================

def load_models(include_variants: bool = False) -> dict:
    """Load pose estimation models. Returns {name: estimator}."""
    models = {}

    # Main models (same as REHAB24-6)
    from src.pose_evaluation.estimators.mediapipe_estimator import MediaPipeEstimator
    from src.pose_evaluation.estimators.movenet_multipose_estimator import MoveNetMultiPoseEstimator
    from src.pose_evaluation.estimators.yolo_estimator import YOLOPoseEstimator

    print("Loading models...")
    models["MediaPipe"] = MediaPipeEstimator(model_complexity=1, min_detection_confidence=0.1)
    print("  MediaPipe Full loaded")

    models["MoveNet"] = MoveNetMultiPoseEstimator()
    print("  MoveNet MultiPose loaded")

    models["YOLOv8"] = YOLOPoseEstimator(model_size="n")
    print("  YOLOv8-Pose Nano loaded")

    if include_variants:
        # MediaPipe variants
        models["MediaPipe_Lite"] = MediaPipeEstimator(model_complexity=0, min_detection_confidence=0.1)
        print("  MediaPipe Lite loaded")
        models["MediaPipe_Heavy"] = MediaPipeEstimator(model_complexity=2, min_detection_confidence=0.1)
        print("  MediaPipe Heavy loaded")

        # YOLO variants
        models["YOLOv8s"] = YOLOPoseEstimator(model_size="s")
        print("  YOLOv8-Pose Small loaded")
        models["YOLOv8m"] = YOLOPoseEstimator(model_size="m")
        print("  YOLOv8-Pose Medium loaded")

    return models


def run_model(estimator, frame: np.ndarray) -> np.ndarray:
    """
    Run model on a single frame.
    Returns: (17, 3) array [x, y, confidence] in COCO format.
    """
    try:
        keypoints = estimator.predict(frame)
        if keypoints is None or len(keypoints) == 0:
            return np.zeros((17, 3))

        # Convert keypoint objects to numpy array
        result = np.zeros((17, 3))
        for kp in keypoints:
            if hasattr(kp, "id") and 0 <= kp.id < 17:
                result[kp.id] = [kp.x, kp.y, kp.confidence]
            elif hasattr(kp, "index") and 0 <= kp.index < 17:
                result[kp.index] = [kp.x, kp.y, kp.confidence]

        # If the above didn't work, try sequential assignment
        if result.sum() == 0 and len(keypoints) >= 17:
            for i, kp in enumerate(keypoints[:17]):
                result[i] = [kp.x, kp.y, kp.confidence]

        return result
    except Exception:
        return np.zeros((17, 3))


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_image(
    pred_17: np.ndarray,
    gt_kps: np.ndarray,
    gt_vis: np.ndarray,
    confidence_threshold: float = 0.5,
) -> dict:
    """Evaluate one prediction against one GT annotation on 12 comparable joints."""

    torso_length = compute_torso_length_coco(gt_kps, gt_vis)
    if np.isnan(torso_length):
        return {"nmpjpe": np.nan, "n_valid_joints": 0, "is_detection_failure": True,
                "per_joint_errors": [np.nan] * 12, "torso_length": np.nan}

    pred_xy = pred_17[COMPARABLE_COCO_INDICES, :2]
    pred_conf = pred_17[COMPARABLE_COCO_INDICES, 2]
    gt_xy = gt_kps[COMPARABLE_COCO_INDICES]
    gt_v = gt_vis[COMPARABLE_COCO_INDICES]

    # Valid = GT visible AND prediction confident
    valid_mask = (gt_v >= 2) & (pred_conf >= confidence_threshold)

    if not np.any(valid_mask) or np.all(pred_xy == 0):
        return {"nmpjpe": np.nan, "n_valid_joints": 0, "is_detection_failure": True,
                "per_joint_errors": [np.nan] * 12, "torso_length": torso_length}

    errors = np.linalg.norm(pred_xy - gt_xy, axis=1)
    normalized = errors / torso_length * 100

    filtered = np.where(valid_mask, normalized, np.nan)
    nmpjpe = float(np.nanmean(filtered))

    return {
        "nmpjpe": nmpjpe,
        "n_valid_joints": int(np.sum(valid_mask)),
        "is_detection_failure": False,
        "per_joint_errors": [float(e) if v else np.nan for e, v in zip(normalized, valid_mask)],
        "torso_length": torso_length,
    }


def match_prediction_to_gt(pred_17: np.ndarray, annotations: list) -> dict:
    """Match prediction to the best GT annotation (by torso proximity)."""
    best_ann = None
    best_dist = float("inf")

    pred_shoulder_center = (pred_17[5, :2] + pred_17[6, :2]) / 2
    pred_hip_center = (pred_17[11, :2] + pred_17[12, :2]) / 2
    pred_torso_center = (pred_shoulder_center + pred_hip_center) / 2

    for ann in annotations:
        gt_kps, gt_vis = get_gt_keypoints(ann)
        if ann["num_keypoints"] < 6:
            continue
        gt_shoulder = (gt_kps[5] + gt_kps[6]) / 2
        gt_hip = (gt_kps[11] + gt_kps[12]) / 2
        gt_center = (gt_shoulder + gt_hip) / 2
        dist = np.linalg.norm(pred_torso_center - gt_center)
        if dist < best_dist:
            best_dist = dist
            best_ann = ann

    return best_ann


# =============================================================================
# Main Pipeline
# =============================================================================

def run_coco_evaluation(
    coco_dir: Path,
    output_dir: Path,
    max_images: int = None,
    include_variants: bool = False,
    confidence_threshold: float = 0.5,
):
    anno_path = coco_dir / "annotations" / "person_keypoints_val2017.json"
    img_dir = coco_dir / "val2017"

    print(f"=== COCO val2017 Evaluation ===")
    print(f"  Images: {img_dir}")
    print(f"  Annotations: {anno_path}")
    print(f"  Confidence: {confidence_threshold}")
    if max_images:
        print(f"  Max images: {max_images} (test mode)")
    print()

    # Load annotations
    images, annos_by_image = load_coco_annotations(anno_path)

    # Filter: images with person annotations that have enough keypoints
    valid_image_ids = []
    for img_id, anns in annos_by_image.items():
        good_anns = [a for a in anns if a["num_keypoints"] >= 6]
        if good_anns:
            valid_image_ids.append(img_id)
    valid_image_ids = sorted(valid_image_ids)

    if max_images:
        valid_image_ids = valid_image_ids[:max_images]

    print(f"Images to evaluate: {len(valid_image_ids)}")

    # Load models
    models = load_models(include_variants)
    print()

    # Process images
    all_rows = []
    t_start = time.time()

    for i, img_id in enumerate(valid_image_ids):
        img_info = images[img_id]
        img_path = img_dir / img_info["file_name"]
        frame = cv2.imread(str(img_path))

        if frame is None:
            continue

        anns = annos_by_image[img_id]
        n_persons = len([a for a in anns if a["num_keypoints"] >= 6])

        # Run each model
        for model_name, estimator in models.items():
            pred = run_model(estimator, frame)

            # Match to best GT annotation
            best_ann = match_prediction_to_gt(pred, anns)
            if best_ann is None:
                continue

            gt_kps, gt_vis = get_gt_keypoints(best_ann)
            result = evaluate_image(pred, gt_kps, gt_vis, confidence_threshold)

            row = {
                "image_id": img_id,
                "model": model_name,
                "nmpjpe": result["nmpjpe"],
                "n_valid_joints": result["n_valid_joints"],
                "is_detection_failure": result["is_detection_failure"],
                "torso_length": result["torso_length"],
                "n_persons_in_image": n_persons,
                "is_single_person": n_persons == 1,
                "gt_num_keypoints": best_ann["num_keypoints"],
                "is_outlier": result["nmpjpe"] > 100 if not np.isnan(result["nmpjpe"]) else False,
            }
            for j, name in enumerate(COMPARABLE_JOINT_NAMES):
                row[f"error_{name}"] = result["per_joint_errors"][j]

            all_rows.append(row)

        if (i + 1) % 100 == 0 or (i + 1) == len(valid_image_ids):
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            eta = (len(valid_image_ids) - i - 1) / fps if fps > 0 else 0
            print(f"  [{i+1}/{len(valid_image_ids)}] {fps:.1f} img/s, ETA {eta/60:.1f} min")

    # Build DataFrame
    df = pd.DataFrame(all_rows)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "coco_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(df):,} rows)")

    # Save metadata
    metadata = {
        "dataset": "COCO val2017",
        "timestamp": datetime.now().isoformat(),
        "n_images": len(valid_image_ids),
        "n_rows": len(df),
        "confidence_threshold": confidence_threshold,
        "models": list(models.keys()),
        "include_variants": include_variants,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print(f"\n=== Results ===")
    valid = df[~df["is_detection_failure"] & ~df["is_outlier"]]
    for model in sorted(valid["model"].unique()):
        m = valid[valid["model"] == model]
        print(f"  {model:20s}: NMPJPE {m['nmpjpe'].mean():.1f}% (median {m['nmpjpe'].median():.1f}%, n={len(m):,})")

    # Single vs multi person
    single = valid[valid["is_single_person"]]
    multi = valid[~valid["is_single_person"]]
    if len(single) > 0 and len(multi) > 0:
        print(f"\n  Single-person images:")
        for model in sorted(single["model"].unique()):
            m = single[single["model"] == model]
            print(f"    {model:20s}: {m['nmpjpe'].mean():.1f}%")
        print(f"  Multi-person images:")
        for model in sorted(multi["model"].unique()):
            m = multi[multi["model"] == model]
            print(f"    {model:20s}: {m['nmpjpe'].mean():.1f}%")

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="COCO val2017 Evaluation")
    parser.add_argument("--test", action="store_true", help="Test mode: 20 images only")
    parser.add_argument("--variants", action="store_true", help="Include model variants")
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    coco_dir = PROJECT_ROOT / "data" / "coco"
    output_dir = PROJECT_ROOT / "evaluation_v2" / "results" / "coco"

    run_coco_evaluation(
        coco_dir=coco_dir,
        output_dir=output_dir,
        max_images=20 if args.test else None,
        include_variants=args.variants,
        confidence_threshold=args.confidence,
    )


if __name__ == "__main__":
    main()
