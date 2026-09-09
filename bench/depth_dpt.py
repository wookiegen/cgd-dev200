"""Depth condition for multigen5k with the pinned annotator Intel/dpt-large (BENCHMARK Section 5).

Writes conditions/depth/<id>.png (8-bit, per-image min-max normalized relative inverse depth, 3-channel; the controller input) and
conditions/depth_raw/<id>.npy (float16, 512x512, the raw DPT prediction resized bicubic; the scorer's reference for scale+shift-aligned MSE/RMSE).
Usage: python depth_dpt.py [--device cuda:0|cpu] [--start 0 --end 5000]
"""
import argparse
import csv
import time

import numpy as np
import torch
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor

from common import OUT_ROOT, REPO_BENCH, env_pins, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=5000)
ap.add_argument("--threads", type=int, default=16)
args = ap.parse_args()
torch.set_num_threads(args.threads)
SET = "multigen5k"
out = OUT_ROOT / SET / "conditions"
(out / "depth").mkdir(parents=True, exist_ok=True)
(out / "depth_raw").mkdir(parents=True, exist_ok=True)
MODEL = "Intel/dpt-large"
proc = DPTImageProcessor.from_pretrained(MODEL)
model = DPTForDepthEstimation.from_pretrained(MODEL).to(args.device).eval()
rows = list(csv.DictReader(open(REPO_BENCH / SET / "manifest.csv")))[args.start:args.end]
t0 = time.time()
with torch.no_grad():
    for k, r in enumerate(rows):
        s = r["sample_id"]
        if (out / "depth_raw" / f"{s}.npy").exists() and (out / "depth" / f"{s}.png").exists():
            continue
        img = Image.open(OUT_ROOT / SET / "images" / f"{s}.png").convert("RGB")
        inp = proc(images=img, return_tensors="pt").to(args.device)
        pred = model(**inp).predicted_depth  # (1, H', W') relative inverse depth
        d = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(512, 512), mode="bicubic", align_corners=False)[0, 0].float().cpu().numpy()
        np.save(out / "depth_raw" / f"{s}.npy", d.astype(np.float16))
        lo, hi = float(d.min()), float(d.max())
        u8 = np.zeros_like(d, dtype=np.uint8) if hi - lo < 1e-8 else ((d - lo) / (hi - lo) * 255.0).round().astype(np.uint8)
        Image.fromarray(np.repeat(u8[:, :, None], 3, axis=2)).save(out / "depth" / f"{s}.png")
        if k % 250 == 0:
            print(f"{k}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
write_json(REPO_BENCH / SET / "DEPTH_VERSION.json", {
    "annotator": MODEL, "processor": "DPTImageProcessor defaults (384x384 input)", "output": "predicted_depth resized bicubic to 512x512",
    "png": "3-channel 8-bit, per-image min-max normalization of the raw prediction (brighter = closer, DPT inverse-depth convention)",
    "raw": "float16 .npy, 512x512, unnormalized; scorer aligns scale+shift per image before MSE/RMSE (BENCHMARK Section 5)",
    "device_used": args.device, "env": env_pins()})
print(f"depth done: {len(rows)} rows in {time.time()-t0:.0f}s on {args.device}")
