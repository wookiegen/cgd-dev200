"""Depth condition of the REAL training crops (for the conditioned VAE decoder archetype): Intel/dpt-large on train/<set>/images512/<sid>.png,
written as train/<set>/conditions_real/depth/<sid>.png (8-bit per-image min-max, 3-channel; the same recipe as the eval conditions in
depth_dpt.py). Skips existing files. Usage: CUDA_VISIBLE_DEVICES=g python precompute_real_depth.py --set multigen_train30 [--shard g --nshards n]
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor

from common import OUT_ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="multigen_train30"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()
T = OUT_ROOT / "train" / a.set
out = T / "conditions_real" / "depth"; out.mkdir(parents=True, exist_ok=True)
files = sorted(glob.glob(str(T / "images512" / "*.png")))
files = [f for i, f in enumerate(files) if i % a.nshards == a.shard and not (out / os.path.basename(f)).exists()]
dev = "cuda"
proc = DPTImageProcessor.from_pretrained("Intel/dpt-large"); dpt = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(dev).eval()
print(f"{a.set} shard {a.shard}/{a.nshards}: {len(files)} crops to do", flush=True)
t0 = time.time()
for i in range(0, len(files), a.batch):
    fs = files[i:i + a.batch]
    ims = [Image.open(f).convert("RGB") for f in fs]
    with torch.no_grad():
        pred = dpt(**proc(images=ims, return_tensors="pt").to(dev)).predicted_depth
        d = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(512, 512), mode="bicubic", align_corners=False)[:, 0].float().cpu().numpy()
    for f, dd in zip(fs, d):
        lo, hi = float(dd.min()), float(dd.max())
        u8 = np.zeros_like(dd, dtype=np.uint8) if hi - lo < 1e-8 else ((dd - lo) / (hi - lo) * 255.0).round().astype(np.uint8)
        Image.fromarray(np.repeat(u8[:, :, None], 3, axis=2)).save(out / os.path.basename(f))
    if (i // a.batch) % 100 == 0:
        print(f"  [{i+len(fs)}/{len(files)}] {time.time()-t0:.0f}s", flush=True)
print(f"done: {len(files)} in {time.time()-t0:.0f}s")
