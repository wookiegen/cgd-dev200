"""Write bench/BENCH.json: version, per-set counts, sha256 of every manifest, source dataset revisions (HF commit at build time), materialized-file counts."""
import csv
import datetime as dt
import os
from pathlib import Path

from common import OUT_ROOT, REPO_BENCH, env_pins, sha256_file, write_json

sets = {}
for s in ["multigen5k", "ade20k_val2k", "coco_val5k", "dreambench750", "train/multigen_train30", "train/ade20k_train", "train/coco_train"]:
    m = REPO_BENCH / s / "manifest.csv"
    if not m.exists():
        continue
    n = sum(1 for _ in csv.DictReader(open(m)))
    entry = {"n": n, "manifest_sha256": sha256_file(m)}
    for extra in ["VERSION.json", "DEPTH_VERSION.json", "palette.json", "class_colors.json", "boxes.jsonl"]:
        if (REPO_BENCH / s / extra).exists():
            entry[extra] = sha256_file(REPO_BENCH / s / extra)
    d = OUT_ROOT / s
    if d.exists():
        entry["materialized"] = {sub: len(os.listdir(d / sub)) for sub in ["images", "labels", "conditions/canny", "conditions/depth", "conditions/depth_raw", "conditions/seg", "conditions/bbox", "refs"] if (d / sub).exists()}
    sets[s] = entry
bl = REPO_BENCH / "blocklist" / "eval_sha256.txt"
revisions = {}
try:
    from huggingface_hub import HfApi
    api = HfApi()
    for rid in ["limingcv/MultiGen-20M_canny_eval", "limingcv/MultiGen-20M_depth_eval", "limingcv/Captioned_ADE20K", "limingcv/MultiGen-20M_train", "google/dreambooth"]:
        try:
            revisions[rid] = api.repo_info(rid, repo_type="dataset").sha
        except Exception as e:  # noqa: BLE001
            revisions[rid] = f"unavailable: {type(e).__name__}"
except Exception:  # noqa: BLE001
    pass
write_json(REPO_BENCH / "BENCH.json", {
    "bench_version": "bench_v1", "built": dt.date.today().isoformat(), "spec": "paper repo docs/BENCHMARK_v1.md v1.7, Section 5c",
    "sample_id": "five-digit, source order (file, then row); latents/outputs key on it", "sha256": "of the ORIGINAL image bytes",
    "crop": "shorter side -> 512, center crop 512x512 (non-square sources only); crop_box in manifests",
    "blocklist": {"file": "blocklist/eval_sha256.txt", "n": sum(1 for l in open(bl) if not l.startswith("#")) if bl.exists() else 0, "rule": "every training loader drops rows whose sha is listed"},
    "source_revisions_at_build": revisions, "sets": sets, "materialized_root": str(OUT_ROOT), "env": env_pins()})
print("BENCH.json written:", {k: v["n"] for k, v in sets.items()})
