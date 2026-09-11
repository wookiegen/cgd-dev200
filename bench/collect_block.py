"""Collect the numbers of one controller block from results/bench/ into a JSON (and a readable print) for filling the paper tables.

Usage: python collect_block.py <controller> [--out /path.json]
Reads <split>/<controller>.<method>@<res>[.<metrics>][.<cond>].json for method in {vae, pid_k28, pid_k24, pid_k16}, res in {512, 2048},
conditions canny / depth (multigen5k) and seg (ade20k_val2k), merging standard, VLM, and adherence-only records.
"""
import argparse
import glob
import json
from pathlib import Path

from common import REPO_BENCH

ap = argparse.ArgumentParser(); ap.add_argument("controller"); ap.add_argument("--out", default=None)
a = ap.parse_args()
R = REPO_BENCH.parent / "results" / "bench"
out = {}
for split, conds in [("multigen5k", ["canny", "depth"]), ("ade20k_val2k", ["seg"]), ("coco_val5k", ["bbox"])]:
    for m in ["vae", "pid_k28", "pid_k24", "pid_k16"]:
        for res in [512, 2048]:
            if m == "vae" and res == 2048:
                continue
            key = f"{m}@{res}"
            for f in sorted(glob.glob(str(R / split / f"{a.controller}.{key}*.json"))):
                for rec in json.load(open(f)):
                    if rec["condition"] not in conds:
                        continue
                    d = out.setdefault(key, {})
                    adh = rec.get("adherence", {}); q = rec.get("quality", {})
                    c = rec["condition"]
                    if c == "canny":
                        d.update({"f1": adh.get("f1"), "f1_strict": adh.get("f1_strict")})
                    elif c == "depth":
                        d.update({"mse": adh.get("mse"), "rmse": adh.get("rmse")})
                    elif c == "seg":
                        d.update({"seg_miou": adh.get("miou")})
                    elif c == "bbox":
                        d.update({"layout_sr": adh.get("sr"), "layout_miou": adh.get("miou")})
                    for k in ["fid", "pfid", "musiq", "niqe", "maniqa", "qalign", "deqa", "unipercept_iaa", "unipercept_iqa", "vqr1", "psnr", "ssim", "lpips", "dists"]:
                        if q.get(k) is not None:
                            d[(k if c != "seg" else f"seg_{k}")] = q[k]
for key in sorted(out):
    print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sorted(out[key].items())})
if a.out:
    json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)
