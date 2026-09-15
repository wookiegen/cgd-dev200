"""Build the CGD subject training manifest from Subjects200K (2026-09-15; the fourth condition's T4 train set).

Subjects200K ships PAIRS: one 1056x528 image holding two 512x512 views of the same item with padding 8. OminiControl's own
subject training (`omini/train_flux/train_subject.py`) keeps only rows whose `quality_assessment` scores compositeStructure,
objectConsistency and imageQuality ALL >= 5, then uses each pair twice, once in each direction. On our copy that filter keeps
119,967 of 206,841 pairs (58.0 percent; a further 26,754 carry no quality dict at all and fail by the same rule).

We do NOT need 120k. The other CGD branches train on 36,122 (canny / depth) and 19,701 (seg) rows, and each row here costs a
2048 PiD decode, so we take a seeded subsample of `--pairs` pairs. We follow OminiControl's convention exactly and use each
selected pair TWICE, once in each direction, so `--pairs 18000` yields 36,000 training rows, matching the canny / depth branch
at the same GPU cost while keeping the released recipe's structure.

Crop convention copied verbatim from OminiControl so our condition matches the distribution its LoRA was trained on:
  left  = (padding, padding, 512 + padding, 512 + padding)                 = (8, 8, 520, 520)
  right = (512 + 2*padding, padding, 1024 + 2*padding, 512 + padding)      = (528, 8, 1040, 520)
  direction 0: left is the TARGET and right the condition, caption = description_0
  direction 1: right is the TARGET and left the condition, caption = description_1

Writes bench/train/subjects200k_train/manifest.csv only. The crops, latents and 2048 targets are produced by
`make_targets.py --set subjects200k_train`, which reads this manifest and re-derives both views from the source bytes.
Usage (inside the container): python build_subjects_train.py [--pairs 18000] [--seed 0] [--val 64]
"""
import argparse
import csv
import glob
import random
from pathlib import Path

import pyarrow.parquet as pq

from common import RAW_ROOT, REPO_BENCH, sha256_bytes, sid

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", type=int, default=18000, help="pairs kept after the quality filter; each yields TWO rows (both directions), so the default gives 36,000")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val", type=int, default=64, help="rows held out as split=val, as in the other training sets")
ap.add_argument("--src", default=str(RAW_ROOT / "subjects200k" / "data"))
a = ap.parse_args()

OUT = REPO_BENCH / "train" / "subjects200k_train"
OUT.mkdir(parents=True, exist_ok=True)
blocked_shas = set(x.strip() for x in open(REPO_BENCH / "blocklist" / "eval_sha256.txt") if x.strip())


def quality_ok(q):
    """OminiControl's filter, reproduced exactly: a missing quality dict fails, and every sub-score must reach 5."""
    if not q:
        return False
    return all((q.get(k) or 0) >= 5 for k in ("compositeStructure", "objectConsistency", "imageQuality"))


files = sorted(glob.glob(str(Path(a.src) / "*.parquet")))
assert files, f"no parquet shards under {a.src}"
cand = []
n_total = n_noq = 0
for fp in files:
    t = pq.read_table(fp, columns=["image", "collection", "quality_assessment", "description"])
    imgs = t.column("image").to_pylist()
    qs = t.column("quality_assessment").to_pylist()
    ds = t.column("description").to_pylist()
    cs = t.column("collection").to_pylist()
    for i, (im, q, d, c) in enumerate(zip(imgs, qs, ds, cs)):
        n_total += 1
        if not q:
            n_noq += 1
        if not quality_ok(q):
            continue
        cand.append({"source_file": Path(fp).name, "source_row": i, "sha256": sha256_bytes(im["bytes"]),
                     "collection": c, "item": (d or {}).get("item", ""), "category": (d or {}).get("category", ""),
                     "d0": (d or {}).get("description_0", ""), "d1": (d or {}).get("description_1", "")})
print(f"scanned {n_total} pairs over {len(files)} shards; {n_noq} without a quality dict; {len(cand)} pass the >=5 filter")

rng = random.Random(a.seed)
rng.shuffle(cand)
keep = cand[: a.pairs]
keep.sort(key=lambda r: (r["source_file"], r["source_row"]))          # manifest stays in SOURCE order, the frozen convention
print(f"keeping {len(keep)} pairs (seed {a.seed})")

rows = []
for r in keep:                                                         # OminiControl's convention: every pair is used in BOTH directions
    for direction in (0, 1):                                           # 0 = left is the target (description_0); 1 = right is the target (description_1)
        blocked = "1" if r["sha256"] in blocked_shas else ""
        rows.append({"sample_id": sid(len(rows)), "source_file": r["source_file"], "source_row": r["source_row"],
                     "sha256": r["sha256"], "direction": direction, "collection": r["collection"],
                     "item": r["item"], "category": r["category"], "caption": r["d1"] if direction else r["d0"],
                     "blocked": blocked, "split": "train"})
n_val = (a.val // 2) * 2          # both directions of a pair are adjacent; keep them on the same side of the split or the
for r in rows[-n_val:]:            # val rows leak their pair's other view into training
    r["split"] = "val"

cols = ["sample_id", "source_file", "source_row", "sha256", "direction", "collection", "item", "category", "caption", "blocked", "split"]
with open(OUT / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
n_blocked = sum(1 for r in rows if r["blocked"])
print(f"wrote {OUT/'manifest.csv'}: {len(rows)} rows from {len(keep)} pairs (each used in both directions), "
      f"{n_blocked} blocked by the eval sha blocklist, {sum(1 for r in rows if r['split']=='val')} val")
