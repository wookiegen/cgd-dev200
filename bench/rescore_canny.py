"""BENCHMARK v1.8 patch: recompute the canny adherence of every existing multigen5k result record (CPU) so that `f1` = tolerant F1 with a
one-condition-pixel tolerance and `f1_strict` = the old pixel-exact F1. Updates <method>@<res>*.json records (condition canny) and the
per-image CSVs in place; skips records that already carry f1_strict unless --force. Usage: python rescore_canny.py [--force] [--only NAME]
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common import OUT_ROOT, REPO_BENCH
from scorers import CannyF1

ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true"); ap.add_argument("--only", default=None)
a = ap.parse_args()
R = REPO_BENCH.parent / "results" / "bench" / "multigen5k"
B = OUT_ROOT / "multigen5k"
rows = list(csv.DictReader(open(REPO_BENCH / "multigen5k" / "manifest.csv")))
conds = {}
sc = CannyF1()

def cond(sid):
    if sid not in conds:
        conds[sid] = np.array(Image.open(B / "conditions" / "canny" / f"{sid}.png").convert("RGB"))
    return conds[sid]

for jf in sorted(glob.glob(str(R / "*@*.json"))):
    if a.only and a.only not in os.path.basename(jf):
        continue
    recs = json.load(open(jf))
    tgt = [r for r in recs if r["condition"] == "canny" and r.get("adherence")]
    if not tgt or (not a.force and "f1_strict" in tgt[0]["adherence"]):
        continue
    rec = tgt[0]; res = rec["res"]; gen = Path(rec["gen_dir"])
    csvf = jf.replace(".json", "_per_image.csv")
    per = list(csv.DictReader(open(csvf))) if os.path.exists(csvf) else [{"sample_id": r["sample_id"]} for r in rows[: rec["n"]]]
    # the 512 view of a 2048-native method: use the _512 sibling dir if present, else downsample
    gen512 = Path(str(gen) + "_512") if (res == 512 and gen.name != "images" and not (gen / f"{per[0]['sample_id']}.png").exists()) else None
    f1s, f1ss = [], []
    for p in per:
        sid = p["sample_id"]
        probe = gen / f"{sid}.png"
        img = np.array(Image.open(probe).convert("RGB"))
        if img.shape[0] != res:
            alt = Path(str(gen).replace(f"pid@{gen.name.split('@')[-1]}", gen.name + "_512")) if "pid@" in gen.name else None
            sib = Path(str(gen).replace("pid_roundtrip/", "pid_roundtrip_512/")) if "pid_roundtrip/" in str(gen) + "/" else None
            for cand in [alt, sib]:
                if cand is not None and (cand / f"{sid}.png").exists():
                    img = np.array(Image.open(cand / f"{sid}.png").convert("RGB")); break
            if img.shape[0] != res:
                img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        s = sc.score(img, cond(sid))
        p["canny_f1"] = s["f1"]; p["canny_f1_strict"] = s["f1_strict"]; f1s.append(s["f1"]); f1ss.append(s["f1_strict"])
    rec["adherence"] = {"f1": float(np.mean(f1s)), "f1_strict": float(np.mean(f1ss)), "tol_px": int(max(1, res // 512))}
    json.dump(recs, open(jf, "w"), indent=1)
    keys = sorted({k for x in per for k in x}, key=lambda k: (k != "sample_id", k))
    with open(csvf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(per)
    print(f"{os.path.basename(jf):40s} res {res}: f1 {rec['adherence']['f1']:.4f}  f1_strict {rec['adherence']['f1_strict']:.4f}  n {len(per)}", flush=True)
print("done")
