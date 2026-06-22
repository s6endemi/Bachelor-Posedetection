"""
COCO val2017 Evaluation — Parallel Version

Runs each model in a separate process for ~3x speedup.

Usage:
    python evaluation_v2/coco_evaluate_parallel.py          # full run, parallel
    python evaluation_v2/coco_evaluate_parallel.py --test    # 50 images test
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")


def create_single_model_script():
    """Create a helper script that evaluates one model on all COCO images."""
    script = PROJECT_ROOT / "evaluation_v2" / "_coco_single_model.py"
    script.write_text(r'''
import argparse, json, sys, time, cv2, numpy as np, pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from evaluation_v2.config import COMPARABLE_COCO_INDICES, COMPARABLE_JOINT_NAMES

def load_model(model_name):
    sys.path.insert(0, str(PROJECT_ROOT))
    if model_name == "MediaPipe":
        from src.pose_evaluation.estimators.mediapipe_estimator import MediaPipeEstimator
        return MediaPipeEstimator(model_complexity=1, min_detection_confidence=0.1)
    elif model_name == "MoveNet":
        from src.pose_evaluation.estimators.movenet_multipose_estimator import MoveNetMultiPoseEstimator
        return MoveNetMultiPoseEstimator()
    elif model_name == "YOLOv8":
        from src.pose_evaluation.estimators.yolo_estimator import YOLOPoseEstimator
        return YOLOPoseEstimator(model_size="n")
    # Variants
    elif model_name == "MediaPipe_Lite":
        from src.pose_evaluation.estimators.mediapipe_estimator import MediaPipeEstimator
        return MediaPipeEstimator(model_complexity=0, min_detection_confidence=0.1)
    elif model_name == "MediaPipe_Heavy":
        from src.pose_evaluation.estimators.mediapipe_estimator import MediaPipeEstimator
        return MediaPipeEstimator(model_complexity=2, min_detection_confidence=0.1)
    elif model_name == "YOLOv8s":
        from src.pose_evaluation.estimators.yolo_estimator import YOLOPoseEstimator
        return YOLOPoseEstimator(model_size="s")
    elif model_name == "YOLOv8m":
        from src.pose_evaluation.estimators.yolo_estimator import YOLOPoseEstimator
        return YOLOPoseEstimator(model_size="m")
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_model(estimator, frame):
    try:
        keypoints = estimator.predict(frame)
        if keypoints is None or len(keypoints) == 0:
            return np.zeros((17, 3))
        result = np.zeros((17, 3))
        for i, kp in enumerate(keypoints[:17]):
            result[i] = [kp.x, kp.y, kp.confidence]
        return result
    except Exception:
        return np.zeros((17, 3))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--coco-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    anno_path = args.coco_dir / "annotations" / "person_keypoints_val2017.json"
    img_dir = args.coco_dir / "val2017"

    with open(anno_path) as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco["images"]}
    annos_by_image = {}
    for ann in coco["annotations"]:
        annos_by_image.setdefault(ann["image_id"], []).append(ann)

    valid_ids = sorted([iid for iid, anns in annos_by_image.items()
                        if any(a["num_keypoints"] >= 6 for a in anns)])
    if args.max_images:
        valid_ids = valid_ids[:args.max_images]

    print(f"[{args.model}] Loading model...")
    estimator = load_model(args.model)
    print(f"[{args.model}] Processing {len(valid_ids)} images...")

    rows = []
    t0 = time.time()
    for i, img_id in enumerate(valid_ids):
        img_info = images[img_id]
        frame = cv2.imread(str(img_dir / img_info["file_name"]))
        if frame is None:
            continue

        anns = [a for a in annos_by_image[img_id] if a["num_keypoints"] >= 6]
        pred = run_model(estimator, frame)

        # Match to best GT by torso proximity
        best_ann = None
        best_dist = float("inf")
        pred_center = ((pred[5,:2] + pred[6,:2]) / 2 + (pred[11,:2] + pred[12,:2]) / 2) / 2
        for ann in anns:
            kps = np.array(ann["keypoints"]).reshape(17, 3)
            gt_center = ((kps[5,:2] + kps[6,:2]) / 2 + (kps[11,:2] + kps[12,:2]) / 2) / 2
            d = np.linalg.norm(pred_center - gt_center)
            if d < best_dist:
                best_dist = d
                best_ann = ann

        if best_ann is None:
            continue

        gt_kps = np.array(best_ann["keypoints"]).reshape(17, 3)
        gt_xy, gt_vis = gt_kps[:, :2], gt_kps[:, 2]

        # Torso length
        if gt_vis[5] < 2 or gt_vis[6] < 2 or gt_vis[11] < 2 or gt_vis[12] < 2:
            continue
        torso = np.linalg.norm((gt_xy[5]+gt_xy[6])/2 - (gt_xy[11]+gt_xy[12])/2)
        if torso < 1:
            continue

        # NMPJPE on 12 comparable joints
        pred_comp = pred[COMPARABLE_COCO_INDICES]
        gt_comp = gt_xy[COMPARABLE_COCO_INDICES]
        vis_comp = gt_vis[COMPARABLE_COCO_INDICES]
        valid_mask = (vis_comp >= 2) & (pred_comp[:, 2] >= args.confidence)

        if not np.any(valid_mask):
            is_fail = True
            nmpjpe = np.nan
        else:
            is_fail = False
            errors = np.linalg.norm(pred_comp[:,:2] - gt_comp, axis=1)
            normed = errors / torso * 100
            filtered = np.where(valid_mask, normed, np.nan)
            nmpjpe = float(np.nanmean(filtered))

        row = {
            "image_id": img_id, "model": args.model, "nmpjpe": nmpjpe,
            "n_valid_joints": int(np.sum(valid_mask)),
            "is_detection_failure": is_fail,
            "is_outlier": nmpjpe > 100 if not np.isnan(nmpjpe) else False,
            "torso_length": torso,
            "n_persons_in_image": len(anns),
            "is_single_person": len(anns) == 1,
        }
        for j, name in enumerate(COMPARABLE_JOINT_NAMES):
            if valid_mask[j]:
                row[f"error_{name}"] = float(normed[j])
            else:
                row[f"error_{name}"] = np.nan
        rows.append(row)

        if (i+1) % 200 == 0:
            elapsed = time.time() - t0
            fps = (i+1) / elapsed
            eta = (len(valid_ids) - i - 1) / fps / 60
            print(f"  [{args.model}] {i+1}/{len(valid_ids)} ({fps:.1f} img/s, ETA {eta:.0f}min)")

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    valid = df[~df["is_detection_failure"] & ~df["is_outlier"]]
    mean = valid["nmpjpe"].mean() if len(valid) > 0 else float("nan")
    print(f"  [{args.model}] Done: {len(valid)} valid, NMPJPE={mean:.1f}%")

if __name__ == "__main__":
    main()
''')
    return script


def run_parallel(models, coco_dir, output_dir, max_images=None, confidence=0.5):
    """Run models in parallel as separate processes."""
    script = create_single_model_script()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== COCO Parallel Evaluation ===")
    print(f"  Models: {models}")
    print(f"  Images: {'all' if not max_images else max_images}")
    print(f"  Running {len(models)} processes in parallel...\n")

    # Launch all processes
    procs = {}
    for model in models:
        out_csv = output_dir / f"coco_{model}.csv"
        cmd = [
            VENV_PYTHON, str(script),
            "--model", model,
            "--coco-dir", str(coco_dir),
            "--output", str(out_csv),
            "--confidence", str(confidence),
        ]
        if max_images:
            cmd.extend(["--max-images", str(max_images)])

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs[model] = {"proc": proc, "csv": out_csv}
        print(f"  Started: {model} (PID {proc.pid})")

    # Wait and collect
    print()
    for model, info in procs.items():
        stdout, stderr = info["proc"].communicate()
        rc = info["proc"].returncode
        if stdout.strip():
            print(stdout.strip())
        if rc != 0:
            print(f"  ERROR [{model}]: exit code {rc}")
            if stderr.strip():
                # Only print last few lines of stderr (skip warnings)
                err_lines = stderr.strip().split("\n")
                for line in err_lines[-5:]:
                    print(f"    {line}")

    # Merge results
    print("\nMerging results...")
    dfs = []
    for model, info in procs.items():
        if info["csv"].exists():
            df = pd.read_csv(info["csv"])
            dfs.append(df)
            print(f"  {model}: {len(df)} rows")

    if not dfs:
        print("  No results!")
        return

    merged = pd.concat(dfs, ignore_index=True)
    merged_path = output_dir / "coco_results.csv"
    merged.to_csv(merged_path, index=False)
    print(f"\nSaved merged: {merged_path} ({len(merged):,} rows)")

    # Summary
    valid = merged[~merged["is_detection_failure"] & ~merged["is_outlier"]]
    print(f"\n=== Results (confidence={confidence}) ===")
    for model in models:
        m = valid[valid["model"] == model]
        if len(m) > 0:
            print(f"  {model:20s}: NMPJPE {m['nmpjpe'].mean():.1f}% (median {m['nmpjpe'].median():.1f}%, n={len(m):,})")

    # Single vs multi
    single = valid[valid["is_single_person"]]
    multi = valid[~valid["is_single_person"]]
    if len(single) > 0:
        print(f"\n  Single-person:")
        for model in models:
            m = single[single["model"] == model]
            if len(m) > 0:
                print(f"    {model:20s}: {m['nmpjpe'].mean():.1f}%")
    if len(multi) > 0:
        print(f"  Multi-person:")
        for model in models:
            m = multi[multi["model"] == model]
            if len(m) > 0:
                print(f"    {model:20s}: {m['nmpjpe'].mean():.1f}%")

    # Save metadata
    metadata = {
        "dataset": "COCO val2017",
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "confidence": confidence,
        "max_images": max_images,
        "total_rows": len(merged),
        "valid_rows": len(valid),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="50 images only")
    parser.add_argument("--variants", action="store_true", help="Include model variants")
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    coco_dir = PROJECT_ROOT / "data" / "coco"
    output_dir = PROJECT_ROOT / "evaluation_v2" / "results" / "coco"

    models = ["MediaPipe", "MoveNet", "YOLOv8"]
    if args.variants:
        models += ["MediaPipe_Lite", "MediaPipe_Heavy",
                    "MoveNet_SP_Lightning", "MoveNet_SP_Thunder",
                    "YOLOv8s", "YOLOv8m"]

    run_parallel(
        models=models,
        coco_dir=coco_dir,
        output_dir=output_dir,
        max_images=50 if args.test else None,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
