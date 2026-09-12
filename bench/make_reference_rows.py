"""Reference rows of tab:main / tab:recon / fig:ceiling (decoder-only, no controller): the VAE round trip and the vanilla PiD round trip.

For every sample of an eval set: encode the real 512 crop with the FLUX VAE -> z0 (cached), decode z0 with the VAE (512) and with vanilla
PiD (2048 + the 512 INTER_AREA view). Outputs under $CGD_BENCH_ROOT/outputs/ref/:
  latents/ref/<split>/<id>.pt                 z0 (scaled space), caption
  outputs/ref/vae_roundtrip/<split>/<id>.png  512
  outputs/ref/pid_roundtrip/<split>/<id>.png  2048 native;  .../pid_roundtrip_512/<split>/<id>.png  matched view
Usage (inside the container, one process per GPU):
  CUDA_VISIBLE_DEVICES=g python make_reference_rows.py --split multigen5k --shard g --nshards 4 [--no-pid] [--limit N]
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from common import OUT_ROOT, REPO_BENCH

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=["multigen5k", "ade20k_val2k", "coco_val5k"])
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--no-pid", action="store_true"); ap.add_argument("--no-vae", action="store_true")
ap.add_argument("--steps", type=int, default=None, help="PiD steps (default 4 student, 25 teacher)"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cfg", type=float, default=None, help="PiD CFG (default 1.0 student, 5.0 teacher)")
ap.add_argument("--pid-ckpt-type", default="2k", help="2k = released distilled student; teacher = undistilled v1.5 (BENCHMARK v1.11)")
ap.add_argument("--out-name", default=None, help="PiD round-trip folder name (default pid_roundtrip; teacher -> pidt_roundtrip)")
ap.add_argument("--subset500", action="store_true", help="only the manifest rows with subset500 == 1")
a = ap.parse_args()
TEACHER = a.pid_ckpt_type == "teacher"
steps = a.steps if a.steps is not None else (25 if TEACHER else 4)
cfg = a.cfg if a.cfg is not None else (5.0 if TEACHER else 1.0)
name = a.out_name or ("pidt_roundtrip" if TEACHER else "pid_roundtrip")

rows = list(csv.DictReader(open(REPO_BENCH / a.split / "manifest.csv")))
if a.subset500:
    rows = [r for r in rows if r.get("subset500") == "1"]
rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard][: a.limit or None]
img_dir = OUT_ROOT / a.split / "images"
d_lat = OUT_ROOT / "latents" / "ref" / a.split
d_vae = OUT_ROOT / "outputs" / "ref" / "vae_roundtrip" / a.split
d_pid = OUT_ROOT / "outputs" / "ref" / name / a.split
d_pid512 = OUT_ROOT / "outputs" / "ref" / f"{name}_512" / a.split
for d in [d_lat, d_vae, d_pid, d_pid512]:
    d.mkdir(parents=True, exist_ok=True)
log = OUT_ROOT / "outputs" / "ref" / f"log_{name}_{a.split}_shard{a.shard}.jsonl"

from flux_pid import FluxVAE, PiD  # noqa: E402  (imports torch/diffusers; PiD chdirs into its root)
vae = FluxVAE()
pid = None if a.no_pid else PiD(ckpt_type=a.pid_ckpt_type)
print(f"{a.split} shard {a.shard}/{a.nshards}: {len(rows)} rows; vae={'yes'} pid={'no' if a.no_pid else pid.experiment} ({steps} steps, cfg {cfg}) -> {name}", flush=True)

t0 = time.time(); n = 0
for r in rows:
    s = r["sample_id"]
    need_vae = not a.no_vae and not (d_vae / f"{s}.png").exists()
    need_pid = not a.no_pid and not (d_pid / f"{s}.png").exists()
    if not need_vae and not need_pid:
        continue
    img = np.array(Image.open(img_dir / f"{s}.png").convert("RGB"))
    lat_p = d_lat / f"{s}.pt"
    if lat_p.exists():
        lat = torch.load(lat_p, map_location="cpu")["latent"]
    else:
        lat = vae.encode([img]).cpu()
        torch.save({"latent": lat, "sigma": 0.0, "caption": r["caption"], "sample_id": s, "split": a.split}, lat_p)
    rec = {"sample_id": s}
    if need_vae:
        t1 = time.time(); y = vae.decode(lat)[0]; rec["t_vae_s"] = time.time() - t1
        Image.fromarray(y).save(d_vae / f"{s}.png")
    if need_pid:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t1 = time.time()
        y = pid.decode(lat, [r["caption"]], sigma=0.0, steps=steps, seed=a.seed, cfg=cfg)[0]
        torch.cuda.synchronize(); rec["t_pid_s"] = time.time() - t1; rec["pid_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        Image.fromarray(y).save(d_pid / f"{s}.png")
        cv2.imwrite(str(d_pid512 / f"{s}.png"), cv2.cvtColor(cv2.resize(y, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
    with open(log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    n += 1
    if n % 50 == 0:
        print(f"[{n}/{len(rows)}] {time.time()-t0:.0f}s ({(time.time()-t0)/n:.2f}s/img)", flush=True)
print(f"done {a.split} shard {a.shard}: {n} new in {time.time()-t0:.0f}s")
