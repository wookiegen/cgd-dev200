"""Native-condition track (user decision 2026-09-12): a 2048 edge condition for every eval image, SYNTHESIZED from the existing 512 data.
The real 512 image is round-tripped through vanilla PiD (already materialized: outputs/ref/pid_roundtrip/<split>/<sid>.png, 2048), and the
2048 condition is Canny(100, 200) of that image: thin edges at native scale, consistent with the 2048 reference by construction.
Writes <split>/conditions/canny2048/<sid>.png (3-channel binary, like conditions/canny). The controllers keep their 512 condition (Canny of
the real image), so every cached latent and decode is reused; only the scoring reference changes (harness.py --cond-res 2048, tol 1 px).
Caveat, stated in the paper: the reference is made by the blind decoder from the real latent, so this track measures recovery of
decoder-consistent native structure specified by a native condition, not agreement with a photograph; no FID against it.
Usage: python build_native_conditions.py [--split multigen5k] [--workers 32]
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from common import OUT_ROOT, REPO_BENCH, env_pins, write_json

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="multigen5k"); ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--src-name", default="pid_roundtrip_s7", help="round-trip folder under outputs/ref used as the 2048 reference (BENCHMARK v1.16: pid_roundtrip_s7 = decoder seed 7, a seed no evaluated row uses, so no row shares the reference's sampler noise; v1.10 to v1.15 used pid_roundtrip = seed 0, kept as conditions/canny2048_seed0)")
ap.add_argument("--out-name", default="canny2048")
a = ap.parse_args()
src = OUT_ROOT / "outputs" / "ref" / a.src_name / a.split
out = OUT_ROOT / a.split / "conditions" / a.out_name; out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(REPO_BENCH / a.split / "manifest.csv")))


def one(sid):
    p = out / f"{sid}.png"
    if p.exists():
        return 0
    img = cv2.imread(str(src / f"{sid}.png"))
    assert img is not None and img.shape[0] == 2048, sid
    e = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
    cv2.imwrite(str(p), np.repeat(e[:, :, None], 3, axis=2))
    return 1


with Pool(a.workers) as pool:
    n = sum(pool.map(one, [r["sample_id"] for r in rows], chunksize=16))
write_json(REPO_BENCH / a.split / "NATIVE_CONDITION_VERSION.json", {
    "condition": a.out_name, "source": f"cv2.Canny(gray, 100, 200) of the vanilla-PiD round trip of the real 512 image (outputs/ref/{a.src_name}, 2048, 4-step 2K distilled checkpoint, sigma 0; seed 7 since BENCHMARK v1.16 = a decoder seed no evaluated row uses, so the reference shares no sampler noise with any row; seed 0 before)",
    "purpose": "native-condition track: the decoder receives the condition at the output resolution; scoring tolerance = one condition pixel = 1 px at 2048",
    "controllers_input": "unchanged: conditions/canny (512, Canny of the real image)", "n": len(rows), "env": env_pins()})
print(f"{a.split}: {n} new canny2048 conditions -> {out}")
