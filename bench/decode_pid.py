"""Decode a controller's cached latents with vanilla PiD (condition-blind, 4-step distilled) at the truncation points: the vanilla-PiD rows.

Reads latents/<controller>/<condition>/<id>.pt (x_0 + xt16/xt24) and writes, per K in --ks:
  outputs/<controller>/<condition>/pid@<K>/<id>.png       2048 native
  outputs/<controller>/<condition>/pid@<K>_512/<id>.png   matched 512 view (cv2.INTER_AREA)
K = 28 decodes x_0 with sigma 0; K = 24 / 16 decode xt<K> with its saved sigma (PiD's sigma-aware adapter).
Usage: CUDA_VISIBLE_DEVICES=g python decode_pid.py --controller omini --condition canny [--ks 28,24,16] [--shard g --nshards n]
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch

from common import OUT_ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--controller", default="omini"); ap.add_argument("--condition", required=True)
ap.add_argument("--ks", default="28,24,16"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--steps", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pid-ckpt-type", default="2k")
a = ap.parse_args()

d_lat = OUT_ROOT / "latents" / a.controller / a.condition
files = sorted(glob.glob(str(d_lat / "*.pt")))
files = [f for i, f in enumerate(files) if i % a.nshards == a.shard][: a.limit or None]
ks = [int(k) for k in a.ks.split(",")]
outs = {K: OUT_ROOT / "outputs" / a.controller / a.condition / f"pid@{K}" for K in ks}
for K, d in outs.items():
    d.mkdir(parents=True, exist_ok=True); (d.parent / f"pid@{K}_512").mkdir(exist_ok=True)
log = OUT_ROOT / "outputs" / a.controller / a.condition / f"log_pid_shard{a.shard}.jsonl"

from flux_pid import PiD  # noqa: E402
pid = PiD(ckpt_type=a.pid_ckpt_type)
print(f"decode {a.controller}/{a.condition}: shard {a.shard}/{a.nshards}, {len(files)} latents, K in {ks}", flush=True)
t0 = time.time(); n = 0
for f in files:
    sid = os.path.splitext(os.path.basename(f))[0]
    todo = [K for K in ks if not (outs[K] / f"{sid}.png").exists()]
    if not todo:
        continue
    d = torch.load(f, map_location="cpu")
    for K in todo:
        lat, sig = (d["latent"], 0.0) if K == d["steps"] else (d[f"xt{K}"], float(d[f"sigma{K}"]))
        torch.cuda.synchronize(); t1 = time.time()
        y = pid.decode(lat, [d["caption"]], sigma=sig, steps=a.steps, seed=a.seed)[0]
        torch.cuda.synchronize(); dt = time.time() - t1
        cv2.imwrite(str(outs[K] / f"{sid}.png"), cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(outs[K].parent / f"pid@{K}_512" / f"{sid}.png"), cv2.cvtColor(cv2.resize(y, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        with open(log, "a") as fl:
            fl.write(json.dumps({"sample_id": sid, "K": K, "sigma": sig, "t_dec_s": dt}) + "\n")
    n += 1
    if n % 50 == 0:
        print(f"[{n}/{len(files)}] {time.time()-t0:.0f}s", flush=True)
print(f"done: {n} latents decoded in {time.time()-t0:.0f}s")
