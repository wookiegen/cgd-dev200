"""Metric sanity for the resolution protocol: score the REAL image, upsampled (bicubic) to 1024 and 2048, against its own 512 canny condition
with the v1.8 scorer (tolerant F1 with tol = R/512 px, strict F1). CPU. Usage: python upsampled_real_f1.py [--n 200|5000]"""
import argparse, csv, sys, json
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import OUT_ROOT, REPO_BENCH
from scorers import CannyF1
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--dev200", action="store_true"); a = ap.parse_args()
rows = list(csv.DictReader(open(REPO_BENCH / "multigen5k" / "manifest.csv")))
if a.dev200:
    seen = set(); rows = [r for r in sorted(rows, key=lambda r: (int(r["dev200_idx"] or 10**6), r["sample_id"])) if r["dev200_idx"] and not (r["dev200_idx"] in seen or seen.add(r["dev200_idx"]))]
rows = rows[: a.n]
sc = CannyF1(); acc = {}
for r in rows:
    sid = r["sample_id"]
    img = np.array(Image.open(OUT_ROOT / "multigen5k" / "images" / f"{sid}.png").convert("RGB"))
    cond = np.array(Image.open(OUT_ROOT / "multigen5k" / "conditions" / "canny" / f"{sid}.png").convert("RGB"))
    for R, interp in [(512, None), (1024, cv2.INTER_CUBIC), (2048, cv2.INTER_CUBIC), (2048, cv2.INTER_LANCZOS4)]:
        up = img if R == 512 else cv2.resize(img, (R, R), interpolation=interp)
        s = sc.score(up, cond); key = f"{R}_{'orig' if R == 512 else ('cubic' if interp == cv2.INTER_CUBIC else 'lanczos')}"
        d = acc.setdefault(key, {"f1": [], "f1_strict": [], "n_edge": []}); d["f1"].append(s["f1"]); d["f1_strict"].append(s["f1_strict"])
        d["n_edge"].append(float(sc.edges(up).mean()))
    # area-downsampled round trip of the upsample (what the matched view of a 2048 output does)
    back = cv2.resize(cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_CUBIC), (512, 512), interpolation=cv2.INTER_AREA)
    s = sc.score(back, cond); d = acc.setdefault("2048cubic_area512", {"f1": [], "f1_strict": [], "n_edge": []}); d["f1"].append(s["f1"]); d["f1_strict"].append(s["f1_strict"]); d["n_edge"].append(float(sc.edges(back).mean()))
out = {k: {m: round(float(np.mean(v)), 4) for m, v in d.items()} for k, d in acc.items()}
out["n"] = len(rows); print(json.dumps(out, indent=1))
