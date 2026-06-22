"""Batch 1: 4 variants parallel on REHAB24-6 (TFLite for MoveNet)."""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
VENV_PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
SCRIPT = str(ROOT / "evaluation_v2" / "rehab_variant_inference.py")

# For TFLite MoveNet, we need a wrapper script
TFLITE_SCRIPT = ROOT / "evaluation_v2" / "_rehab_movenet_tflite.py"
TFLITE_SCRIPT.write_text(r'''
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
''')

models = [
    # (name, command)
    ("MediaPipe_Lite", [VENV_PY, SCRIPT, "--model", "MediaPipe_Lite"]),
    ("MoveNet_SP_Lightning", [VENV_PY, str(TFLITE_SCRIPT),
        "--model", "MoveNet_SP_Lightning",
        "--tflite-path", str(ROOT / "models" / "movenet_singlepose_lightning_fp16.tflite"),
        "--input-size", "192"]),
    ("MoveNet_SP_Thunder", [VENV_PY, str(TFLITE_SCRIPT),
        "--model", "MoveNet_SP_Thunder",
        "--tflite-path", str(ROOT / "models" / "movenet_singlepose_thunder_fp16.tflite"),
        "--input-size", "256"]),
    ("YOLOv8s", [VENV_PY, SCRIPT, "--model", "YOLOv8s"]),
]

print("=== BATCH 1: 4 variants parallel ===")
print(f"  {', '.join(n for n,_ in models)}")
print()

procs = {}
for name, cmd in models:
    log = ROOT / "evaluation_v2" / "results" / f"log_{name}.txt"
    f = open(log, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    procs[name] = {"proc": p, "log": log, "file": f, "start": time.time()}
    print(f"  Started {name} (PID {p.pid})")

print(f"\nAll 4 running. Monitoring...")

while any(p["proc"].poll() is None for p in procs.values()):
    time.sleep(60)
    for name, info in procs.items():
        status = "RUNNING" if info["proc"].poll() is None else f"DONE (exit {info['proc'].returncode})"
        elapsed = (time.time() - info["start"]) / 60
        print(f"  [{elapsed:5.0f}min] {name:25s}: {status}")
    print()

for info in procs.values():
    info["file"].close()

print("=== BATCH 1 COMPLETE ===")
for name, info in procs.items():
    elapsed = (time.time() - info["start"]) / 60
    rc = info["proc"].returncode
    print(f"  {name:25s}: exit={rc}, {elapsed:.0f} min")
