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
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--steps", type=int, default=None, help="decoder steps (default: 4 for the distilled student, 25 for the teacher)")
ap.add_argument("--cfg", type=float, default=None, help="decoder CFG (default: 1.0 student, 5.0 teacher, PiD's documented settings)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pid-ckpt-type", default="2k", help="2k = released 4-step distilled student (main tables); teacher = PiD_v1pt5 undistilled (BENCHMARK v1.11)")
ap.add_argument("--out-name", default=None, help="output folder prefix (default: pid for the student, pidt for the teacher) -> outputs/<ctrl>/<cond>/<out-name>@<K>")
ap.add_argument("--subset500", action="store_true", help="decode only the manifest rows with subset500 == 1 (multigen5k)")
a = ap.parse_args()
TEACHER = a.pid_ckpt_type == "teacher"
steps = a.steps if a.steps is not None else (25 if TEACHER else 4)
cfg = a.cfg if a.cfg is not None else (5.0 if TEACHER else 1.0)
name = a.out_name or ("pidt" if TEACHER else "pid")

d_lat = OUT_ROOT / "latents" / a.controller / a.condition
files = sorted(glob.glob(str(d_lat / "*.pt")))
if a.subset500:
    import csv
    from common import REPO_BENCH
    keep = {r["sample_id"] for r in csv.DictReader(open(REPO_BENCH / "multigen5k" / "manifest.csv")) if r.get("subset500") == "1"}
    files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in keep]
files = [f for i, f in enumerate(files) if i % a.nshards == a.shard][: a.limit or None]
ks = [int(k) for k in a.ks.split(",")]
outs = {K: OUT_ROOT / "outputs" / a.controller / a.condition / f"{name}@{K}" for K in ks}
for K, d in outs.items():
    d.mkdir(parents=True, exist_ok=True); (d.parent / f"{name}@{K}_512").mkdir(exist_ok=True)
log = OUT_ROOT / "outputs" / a.controller / a.condition / f"log_{name}_shard{a.shard}.jsonl"

from flux_pid import PiD  # noqa: E402
pid = PiD(ckpt_type=a.pid_ckpt_type)
print(f"decoder {name}: ckpt {a.pid_ckpt_type}, {steps} steps, cfg {cfg}", flush=True)
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
        y = pid.decode(lat, [d["caption"]], sigma=sig, steps=steps, seed=a.seed, cfg=cfg)[0]
        torch.cuda.synchronize(); dt = time.time() - t1
        cv2.imwrite(str(outs[K] / f"{sid}.png"), cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(outs[K].parent / f"{name}@{K}_512" / f"{sid}.png"), cv2.cvtColor(cv2.resize(y, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        with open(log, "a") as fl:
            fl.write(json.dumps({"sample_id": sid, "K": K, "sigma": sig, "t_dec_s": dt, "decoder": name, "steps": steps, "cfg": cfg}) + "\n")
    n += 1
    if n % 50 == 0:
        print(f"[{n}/{len(files)}] {time.time()-t0:.0f}s", flush=True)
print(f"done: {n} latents decoded in {time.time()-t0:.0f}s")
