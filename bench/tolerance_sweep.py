"""Edge F1 as a function of the matching tolerance, at the 512 matched view (2026-09-15).

Why: BENCHMARK v1.8 replaced the pixel-exact edge F1 with a tolerant F1 whose tolerance is ONE CONDITION PIXEL, which is 1 px
at the 512 view and 4 px at 2048. The stated reason was the resolution mismatch at 2048 (a 4-px-thick nearest-upsampled
reference against 1-2 px re-extracted edges). At 512 there is no mismatch, so the choice of tolerance 1 rather than 0 is a
free parameter, and it is not neutral across methods: a decoder that repositions the edges it already has (the conditioned
deterministic archetypes) gains on the pixel-exact score and nothing on the tolerant one. This sweeps the tolerance from 0 to
4 px on the same cached 512 outputs so the paper can show that the ranking is stable wherever the tolerance is positive, and
say exactly which part of each method's agreement lives inside the first pixel.

Everything here is CPU-only and reads cached outputs; no decode, no GPU. Usage (inside the container):
  python tolerance_sweep.py [--limit N] [--workers 64]
Writes results/bench/metric_checks/tolerance_sweep_multigen5k.json and prints the table.
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from common import OUT_ROOT, REPO_BENCH
from scorers import CannyF1

BENCH = OUT_ROOT / "multigen5k"
OUTS = OUT_ROOT / "outputs"
# Every row is 512-native or the cached INTER_AREA 512 view of a 2048 output, so no row is resampled here.
ROWS = [
    ("real image", BENCH / "images"),
    ("VAE round trip", OUTS / "ref" / "vae_roundtrip" / "multigen5k"),
    ("PiD round trip", OUTS / "ref" / "pid_roundtrip_512" / "multigen5k"),
    ("OminiControl + VAE decode", OUTS / "omini" / "canny" / "vae@28"),
    ("OminiControl + PiD K=28", OUTS / "omini" / "canny" / "pid@28_512"),
    ("OminiControl + PiD K=24", OUTS / "omini" / "canny" / "pid@24_512"),
    ("OminiControl + PiD K=16", OUTS / "omini" / "canny" / "pid@16_512"),
    ("OminiControl + cond. VAE, additive", OUTS / "omini" / "canny" / "condvae"),
    ("OminiControl + cond. VAE, modulated", OUTS / "omini" / "canny" / "condvae_mod"),
]
TOLS = [0, 1, 2, 3, 4]

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=64)
ap.add_argument("--out", default=str(REPO_BENCH.parent / "results" / "bench" / "metric_checks" / "tolerance_sweep_multigen5k.json"))
a = ap.parse_args()


def one(sid):
    """Return {label: [f1 at each tolerance]} for one sample, or None if any row is missing the image."""
    cond = cv2.imread(str(BENCH / "conditions" / "canny" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
    ref = cond > 0
    out = {}
    for label, d in ROWS:
        img = cv2.imread(str(d / f"{sid}.png"), cv2.IMREAD_COLOR)
        if img is None:
            return None
        assert img.shape[0] == 512, (label, sid, img.shape)
        pred = CannyF1.edges(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        f1s = []
        for t in TOLS:
            p, r = CannyF1.pr_tolerant(pred, ref, t)
            f1s.append(2 * p * r / max(p + r, 1e-9))
        out[label] = f1s
    return out


if __name__ == "__main__":
    sids = [r["sample_id"] for r in csv.DictReader(open(REPO_BENCH / "multigen5k" / "manifest.csv"))][: a.limit or None]
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap_unordered(one, sids, chunksize=8) if r is not None]
    means = {label: [float(np.mean([r[label][i] for r in res])) for i in range(len(TOLS))] for label, _ in ROWS}
    rec = {"split": "multigen5k", "view": 512, "cond_res": 512, "tolerances_px": TOLS, "n": len(res),
           "note": "canny F1 at the 512 matched view against conditions/canny; tol 0 = the pixel-exact F1 of the standard "
                   "protocol (f1_strict), tol 1 = the paper metric (one condition pixel at 512)",
           "rows": {label: {"f1": means[label], "dir": str(d)} for label, d in ROWS}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    w = max(len(l) for l, _ in ROWS)
    print(f"n = {len(res)}\n{'row'.ljust(w)} " + "  ".join(f"tol{t}" .rjust(6) for t in TOLS) + "   tol1-tol0")
    for label, _ in ROWS:
        m = means[label]
        print(f"{label.ljust(w)} " + "  ".join(f"{v:6.4f}" for v in m) + f"   {m[1] - m[0]:+.4f}")
    print("wrote", a.out)
