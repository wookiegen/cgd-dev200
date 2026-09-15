"""Backfill the 2048 condition-map provenance into canny records written before the stamp existed (2026-09-15).

Why: BENCHMARK v1.16 replaced the c2048 reference (a PiD round trip at decoder seed 0) with one at held-out seed 7 and
rescored everything under results/bench/, but results/variants/ was never rescored and no record recorded WHICH map it was
scored against. `eval_variant.py` then compared a stale seed-0 variant with a rescored seed-7 baseline and reported a
+0.0282 win, CI excluding zero, for a decoder that was a byte-identical copy of the baseline.

`harness.py` now stamps `cond_ref` on every new canny record. This backfills the same stamp onto the existing ones, which is
what lets `eval_variant.guard_same_reference` abort on an unstamped (therefore pre-v1.16, therefore seed-0) variant record:
  results/bench/**              -> cond_ref "seed7"   (all rescored by the v1.16 fix; verified by four gates at the time)
  results/bench/_archive_c2048_seed0/** -> cond_ref "seed0"   (the retired map, kept only as evidence)
Only records whose adherence carries cond_res == 2048 are touched; nothing else in the file changes.

Usage (inside the container): python stamp_cond_provenance.py [--dry-run]
"""
import argparse
import json
from pathlib import Path

from common import REPO_BENCH

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

ROOT = REPO_BENCH.parent / "results"
NOTE = ("BENCHMARK v1.16: the 2048 reference is a PiD round trip at a held-out decoder seed; "
        "records built on different seeds are NOT comparable")

n_stamped = n_skipped = 0
for f in sorted(ROOT.rglob("*.json")):
    if "cond2048" not in f.name:
        continue
    if "/variants/" in str(f).replace("\\", "/"):
        continue          # NEVER stamp a variant: it was not rescored for v1.16, and an absent stamp is the signal that stops the comparison
    try:
        recs = json.load(open(f))
    except Exception:
        continue
    if not isinstance(recs, list):
        continue
    seed = "seed0" if "_archive_c2048_seed0" in str(f) else "seed7"
    changed = False
    for r in recs:
        adh = r.get("adherence") or {}
        if adh.get("cond_res") != 2048 or "cond_ref" in adh:
            continue
        adh["cond_src"] = "conditions/canny2048"
        adh["cond_ref"] = seed
        adh["cond_ref_note"] = NOTE
        changed = True
    if not changed:
        n_skipped += 1
        continue
    n_stamped += 1
    print(f"  {seed}  {f.relative_to(ROOT)}")
    if not a.dry_run:
        json.dump(recs, open(f, "w"), indent=1)

print(f"\n{'would stamp' if a.dry_run else 'stamped'} {n_stamped} records; {n_skipped} already stamped or not applicable")
print("NOTE: results/variants/ is deliberately NOT stamped: those records were never rescored for v1.16, so leaving them "
      "unstamped is what makes eval_variant refuse to compare them.")
