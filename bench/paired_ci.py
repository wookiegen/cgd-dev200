"""Paired bootstrap 95% CIs for the difference between two result records on the same images (per-image CSVs of the harness), and the
effect of excluding the dev-200 images from the multigen5k means. CPU, no models.
Usage: python paired_ci.py --a results/bench/multigen5k/omini.vae@512.canny_per_image.csv --b results/bench/multigen5k/omini.pid_k24@512.canny_per_image.csv --col canny_f1
       python paired_ci.py --exclude-dev200 results/bench/multigen5k/omini.pid_k24@512.canny_per_image.csv --col canny_f1
"""
import argparse
import csv

import numpy as np

from common import REPO_BENCH

ap = argparse.ArgumentParser()
ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--col", default="canny_f1"); ap.add_argument("--n-boot", type=int, default=10000)
ap.add_argument("--exclude-dev200", default=None, help="a per-image CSV: report its mean with and without the dev-200 ids")
a = ap.parse_args()
rng = np.random.default_rng(0)


def load(p):
    return {r["sample_id"]: float(r[a.col]) for r in csv.DictReader(open(p)) if r.get(a.col) not in (None, "", "nan")}


if a.exclude_dev200:
    dev = {r["sample_id"] for r in csv.DictReader(open(REPO_BENCH / "multigen5k" / "manifest.csv")) if r["dev200_idx"]}
    d = load(a.exclude_dev200); allv = np.array(list(d.values())); rest = np.array([v for k, v in d.items() if k not in dev])
    print(f"{a.col}: all {len(allv)} = {allv.mean():.4f}; without dev-200 ({len(rest)}) = {rest.mean():.4f}; delta {rest.mean()-allv.mean():+.4f}")
else:
    A, B = load(a.a), load(a.b); ids = sorted(set(A) & set(B))
    x = np.array([A[i] for i in ids]); y = np.array([B[i] for i in ids]); d = y - x
    idx = rng.integers(0, len(d), size=(a.n_boot, len(d))); boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"n {len(d)}  mean A {x.mean():.4f}  mean B {y.mean():.4f}  B-A {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  B>A on {int((d>0).sum())} images")
