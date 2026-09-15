"""Where each decoder's edge error lives: existence versus localization, at the 512 matched view (2026-09-15).

Companion to tolerance_sweep.py, written to answer three questions the paper has to answer about reporting two edge scores:
  1. why the conditioned deterministic decoder gains on the pixel-exact F1 and nothing on the tolerant one;
  2. whether the pixel-exact score is neutral between a deterministic and a generative decoder;
  3. how much of the pixel-exact score is reproducible at all, i.e. how much of it a decoder's own sampler noise moves.

It reports, per row at 512: precision, recall and F1 at tolerance 0 and 1, and the edge density of the output. Reading:
tolerant scores existence (does an edge exist within one pixel), the tol-0 minus tol-1 gap is localization, and recall
versus precision says whether a change came from hitting reference edges or from not emitting unreferenced ones. The
controlled pairs are (a) the two round trips, which receive the IDENTICAL latent of a real image and differ only in whether
the decoder is deterministic or generative, and (b) the three decoder seeds of one latent, which differ only in sampler noise.

CPU only, reads cached 512 outputs. Usage (inside the container): python edge_error_decomposition.py [--workers 128]
Writes results/bench/metric_checks/edge_error_decomposition_multigen5k.json.
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
ROWS = [
    # (a) same real-image latent, deterministic vs generative decoder
    ("VAE round trip (det.)", OUTS / "ref" / "vae_roundtrip" / "multigen5k"),
    ("PiD round trip (gen.)", OUTS / "ref" / "pid_roundtrip_512" / "multigen5k"),
    # the OminiControl latent, the four decoders of the two-by-two
    ("Omini + VAE decode", OUTS / "omini" / "canny" / "vae@28"),
    ("Omini + PiD K=28", OUTS / "omini" / "canny" / "pid@28_512"),
    ("Omini + cond. VAE, additive", OUTS / "omini" / "canny" / "condvae"),
    ("Omini + cond. VAE, modulated", OUTS / "omini" / "canny" / "condvae_mod"),
    # (b) one latent, three decoder seeds: how much of each score is sampler noise
    ("Omini + PiD K=24, seed 0", OUTS / "omini" / "canny" / "pid@24_512"),
    ("Omini + PiD K=24, seed 1", OUTS / "omini" / "canny" / "pid_s1@24_512"),
    ("Omini + PiD K=24, seed 2", OUTS / "omini" / "canny" / "pid_s2@24_512"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=128)
ap.add_argument("--out", default=str(REPO_BENCH.parent / "results" / "bench" / "metric_checks" / "edge_error_decomposition_multigen5k.json"))
a = ap.parse_args()


def one(sid):
    """Rows are scored independently: the three decoder-seed runs exist on the 500-image subset only, and must not cap the rest."""
    ref = cv2.imread(str(BENCH / "conditions" / "canny" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE) > 0
    out = {}
    for label, d in ROWS:
        img = cv2.imread(str(d / f"{sid}.png"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        pred = CannyF1.edges(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        rec = {"density": float(pred.mean())}
        for t in (0, 1):
            p, r = CannyF1.pr_tolerant(pred, ref, t)
            rec[f"p{t}"], rec[f"r{t}"] = p, r
            rec[f"f{t}"] = 2 * p * r / max(p + r, 1e-9)
        out[label] = rec
    return out


if __name__ == "__main__":
    sids = [r["sample_id"] for r in csv.DictReader(open(REPO_BENCH / "multigen5k" / "manifest.csv"))][: a.limit or None]
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap_unordered(one, sids, chunksize=8) if r is not None]
    keys = ["p0", "r0", "f0", "p1", "r1", "f1", "density"]
    means = {label: {**{k: float(np.mean([r[label][k] for r in res if label in r])) for k in keys},
                     "n": sum(label in r for r in res)} for label, _ in ROWS}
    json.dump({"split": "multigen5k", "view": 512, "rows": means}, open(a.out, "w"), indent=1)
    w = max(len(l) for l, _ in ROWS)
    print(f"{'row'.ljust(w)}     n     P0     R0     F0  |    P1     R1     F1  | F1-F0  density")
    for label, _ in ROWS:
        m = means[label]
        print(f"{label.ljust(w)} {m['n']:5d} {m['p0']:.4f} {m['r0']:.4f} {m['f0']:.4f} | {m['p1']:.4f} {m['r1']:.4f} {m['f1']:.4f} | "
              f"{m['f1'] - m['f0']:+.4f}  {m['density']:.4f}")
    print("wrote", a.out)
