
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
    elif model_name == "MoveNet_SP_Lightning":
        from src.pose_evaluation.estimators.movenet_estimator import MoveNetEstimator
        return MoveNetEstimator(model_name="lightning")
    elif model_name == "MoveNet_SP_Thunder":
        from src.pose_evaluation.estimators.movenet_estimator import MoveNetEstimator
        return MoveNetEstimator(model_name="thunder")
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
