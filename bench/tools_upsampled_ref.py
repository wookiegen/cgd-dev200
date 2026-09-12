"""Indirect native-resolution reference for 512-native rows (Real image, VAE round trip, each controller's VAE decode): the 512 output is
upsampled to 2048 with a bicubic filter and scored with the v1.8 canny scorer at 2048 (tolerance 4 px = one condition pixel; strict kept).
This is NOT a native output and the harness refuses to upsample by design; the value is a marked REFERENCE (paper symbol $^{\\ddag}$): the
score of the interpolation route to 2048, a lower bound on what native detail consistent with the same 512 content would reach.
Writes results/bench/multigen5k/<controller.>method@2048ref.canny.json (+ _per_image.csv). CPU. Usage:
  python tools_upsampled_ref.py --method real --gen-dir $CGD_BENCH_ROOT/multigen5k/images
  python tools_upsampled_ref.py --method vae --controller omini --gen-dir $CGD_BENCH_ROOT/outputs/omini/canny/vae@28
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common import OUT_ROOT, REPO_BENCH
from scorers import CannyF1

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--controller", default=""); ap.add_argument("--gen-dir", required=True)
ap.add_argument("--res", type=int, default=2048); ap.add_argument("--interp", default="cubic", choices=["cubic", "linear", "lanczos"])
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
INTERP = {"cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR, "lanczos": cv2.INTER_LANCZOS4}[a.interp]
rows = list(csv.DictReader(open(REPO_BENCH / "multigen5k" / "manifest.csv")))[: a.limit or None]
gen = Path(a.gen_dir); cond_dir = OUT_ROOT / "multigen5k" / "conditions" / "canny"
out_dir = REPO_BENCH.parent / "results" / "bench" / "multigen5k"; out_dir.mkdir(parents=True, exist_ok=True)
tag = (f"{a.controller}." if a.controller else "") + f"{a.method}@{a.res}ref"
sc = CannyF1(); per = []; t0 = time.time()
for k, r in enumerate(rows):
    sid = r["sample_id"]
    img = np.array(Image.open(gen / f"{sid}.png").convert("RGB"))
    assert img.shape[0] == 512, (sid, img.shape)
    up = cv2.resize(img, (a.res, a.res), interpolation=INTERP)
    s = sc.score(up, np.array(Image.open(cond_dir / f"{sid}.png").convert("RGB")))
    per.append({"sample_id": sid, "canny_f1": s["f1"], "canny_f1_strict": s["f1_strict"]})
    if (k + 1) % 500 == 0:
        print(f"  [{k+1}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)
rec = {"method": a.method, "controller": a.controller, "condition": "canny", "split": "multigen5k", "res": a.res, "native_res": 512,
       "via": f"{a.interp}_x{a.res // 512}_upsample_REFERENCE_not_native", "adherence": {"f1": float(np.mean([p["canny_f1"] for p in per])),
       "f1_strict": float(np.mean([p["canny_f1_strict"] for p in per])), "tol_px": a.res // 512}, "n": len(per), "gen_dir": str(gen)}
json.dump([rec], open(out_dir / f"{tag}.canny.json", "w"), indent=1)
with open(out_dir / f"{tag}.canny_per_image.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sample_id", "canny_f1", "canny_f1_strict"]); w.writeheader(); w.writerows(per)
print(json.dumps({k: rec[k] for k in ["method", "controller", "res", "via", "adherence", "n"]}))
