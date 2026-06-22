
import argparse, sys, time, cv2, numpy as np
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import tensorflow as tf
from src.pose_evaluation.estimators.base import Keypoint
from evaluation_v2.rehab_variant_inference import find_videos, process_video

class MoveNetSPTFLite:
    def __init__(self, model_path, input_size):
        self.interp = tf.lite.Interpreter(model_path=str(model_path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()
        self.out = self.interp.get_output_details()
        self.size = input_size
    def predict(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.size, self.size))
        input_data = np.expand_dims(resized.astype(np.uint8), axis=0)
        self.interp.set_tensor(self.inp[0]['index'], input_data)
        self.interp.invoke()
        kps = self.interp.get_tensor(self.out[0]['index'])[0, 0]
        names = ['nose','left_eye','right_eye','left_ear','right_ear',
                 'left_shoulder','right_shoulder','left_elbow','right_elbow',
                 'left_wrist','right_wrist','left_hip','right_hip',
                 'left_knee','right_knee','left_ankle','right_ankle']
        return [Keypoint(x=float(x*w), y=float(y*h), confidence=float(c), name=names[i])
                for i, (y,x,c) in enumerate(kps)]

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--tflite-path", required=True)
parser.add_argument("--input-size", type=int, required=True)
args = parser.parse_args()

print(f"[{args.model}] Loading TFLite model...")
estimator = MoveNetSPTFLite(args.tflite_path, args.input_size)

videos = find_videos(Path(ROOT / "data"))
out_dir = Path(ROOT / "data" / "predictions_variants" / args.model)
print(f"[{args.model}] Processing {len(videos)} videos...")

t0 = time.time()
for i, video in enumerate(videos):
    result = process_video(video, estimator, args.model, out_dir)
    if (i+1) % 20 == 0 or (i+1) == len(videos):
        elapsed = time.time() - t0
        vps = (i+1) / elapsed
        eta = (len(videos)-i-1) / vps / 60
        print(f"  [{args.model}] {i+1}/{len(videos)} | {result['fps']:.0f} FPS | ETA {eta:.0f}min")

elapsed = time.time() - t0
print(f"[{args.model}] DONE in {elapsed/60:.1f} min")
