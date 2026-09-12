#!/usr/bin/env python
"""02b: Decode the dev-200 cached latents with the undistilled PiD TEACHER (BENCHMARK v1.11), the checkpoint CGD trains on.

Same latents as 02 (`latents/omini_canny/{idx}.pt`: x_0, xt24, xt16), PiD's documented teacher setting (25 steps, CFG 5), 512 -> 2048.
  outputs/omini_pidt/{idx}.png           final latent, 2048;   outputs/omini_pidt_512/{idx}.png      INTER_AREA view
  outputs/omini_pidt_et24/{idx}.png      x_t after 24/28;      outputs/omini_pidt_et16/{idx}.png     x_t after 16/28   (+ _512 views)
~25 s per decode on one H200 (20 GB). Usage: CUDA_VISIBLE_DEVICES=g python scripts/02b_decode_teacher.py [--variants final,et24,et16] [--shard g --nshards n]
"""
import argparse, glob, json, os, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "bench"))
import cv2, numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=REPO)
ap.add_argument("--variants", default="final,et24,et16")
ap.add_argument("--steps", type=int, default=25); ap.add_argument("--cfg", type=float, default=5.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.repo, "latents", "omini_canny", "*.pt")))
files = [f for i, f in enumerate(files) if i % a.nshards == a.shard][: a.limit or None]
variants = a.variants.split(",")
outs = {v: os.path.join(a.repo, "outputs", "omini_pidt" + ("" if v == "final" else f"_{v}")) for v in variants}
for d in outs.values():
    os.makedirs(d, exist_ok=True); os.makedirs(d + "_512", exist_ok=True)
log = os.path.join(a.repo, "results", "log_02b_teacher.jsonl"); os.makedirs(os.path.dirname(log), exist_ok=True)

from flux_pid import PiD  # noqa: E402  (chdir's into the PiD root; all paths here are absolute)
pid = PiD(ckpt_type="teacher")
print(f"teacher {pid.experiment}: {len(files)} latents, variants {variants}, {a.steps} steps, cfg {a.cfg}", flush=True)
t0 = time.time(); n = 0
for f in files:
    idx = os.path.splitext(os.path.basename(f))[0]
    todo = [v for v in variants if a.overwrite or not os.path.exists(os.path.join(outs[v], f"{idx}.png"))]
    if not todo:
        continue
    d = torch.load(f, map_location="cpu", weights_only=False)
    for v in todo:
        if v == "final":
            lat, sig = d["latent"], 0.0
        else:
            K = int(v[2:]); lat, sig = d[f"xt{K}"], float(d[f"sigma{K}"])
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t1 = time.time()
        y = pid.decode(lat, [d["caption"]], sigma=sig, steps=a.steps, seed=a.seed, cfg=a.cfg)[0]
        torch.cuda.synchronize(); dt = time.time() - t1; mem = torch.cuda.max_memory_allocated() / 1e9
        Image.fromarray(y).save(os.path.join(outs[v], f"{idx}.png"))
        cv2.imwrite(os.path.join(outs[v] + "_512", f"{idx}.png"), cv2.cvtColor(cv2.resize(y, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        with open(log, "a") as fl:
            fl.write(json.dumps({"idx": idx, "variant": v, "sigma": sig, "t_dec_s": dt, "peak_mem_gb": mem, "steps": a.steps, "cfg": a.cfg, "decoder": "teacher"}) + "\n")
    n += 1
    if n % 20 == 0:
        print(f"[{n}/{len(files)}] {time.time()-t0:.0f}s", flush=True)
print(f"done shard {a.shard}: {n} latents in {time.time()-t0:.0f}s")
