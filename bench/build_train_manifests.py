"""Training manifests + the leakage blocklist (BENCHMARK v1.7, Section 5c).

1. blocklist/eval_sha256.txt = union of the sha256 of every eval image (multigen5k, ade20k_val2k, coco_val5k, dreambench750 refs).
2. train/<set>/manifest.csv for multigen_train30 (the 30 downloaded MultiGen-20M train shards), ade20k_train (Captioned_ADE20K train),
   coco_train (COCO 2017 train2017 + instances/captions), each with a `blocked` column (1 = sha in the blocklist -> the loader MUST drop it).
3. A fixed held-out VALIDATION split inside each training set (`split` column: 'train' / 'val'; 500 non-blocked rows per set,
   numpy.random.default_rng(<VAL_SEED>).choice), for early stopping and hyperparameters, so the eval sets are never used for tuning.
Run AFTER the four eval builders.
"""
import numpy as np

VAL_SEED = 1000
VAL_N = 500


def assign_split(rows):
    ok = [i for i, r in enumerate(rows) if not r["blocked"]]
    pick = set(np.random.default_rng(VAL_SEED).choice(ok, min(VAL_N, len(ok)), replace=False).tolist())
    for i, r in enumerate(rows):
        r["split"] = "val" if i in pick else "train"
    return sum(1 for r in rows if r["split"] == "val")
import csv
import glob
import io
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

from common import RAW_ROOT, REPO_BENCH, crop_params, env_pins, parquet_rows, sha256_bytes, sha256_file, sid, write_json, write_manifest

# ------------------------------------------------------------- blocklist
block = {}
for s, col in [("multigen5k", "sha256"), ("ade20k_val2k", "sha256"), ("coco_val5k", "sha256"), ("dreambench750", "ref_sha256")]:
    for r in csv.DictReader(open(REPO_BENCH / s / "manifest.csv")):
        block.setdefault(r[col], []).append(f"{s}:{r['sample_id']}")
(REPO_BENCH / "blocklist").mkdir(exist_ok=True)
with open(REPO_BENCH / "blocklist" / "eval_sha256.txt", "w") as f:
    f.write("# sha256 of every CGD benchmark eval image (original bytes). Every training loader must drop rows whose image sha is listed here.\n# sha256\teval_rows\n")
    for h in sorted(block):
        f.write(f"{h}\t{','.join(block[h])}\n")
print(f"blocklist: {len(block)} unique eval shas")

# ------------------------------------------------------------- multigen_train30
files = sorted(glob.glob(str(RAW_ROOT / "multigen_train_subset" / "data" / "train-*.parquet")))
rows, nb = [], 0
for i, (fname, row, r) in enumerate(parquet_rows(files, ["image", "text"])):
    h = sha256_bytes(r["image"]["bytes"])
    b = int(h in block); nb += b
    rows.append({"sample_id": sid(i), "source_file": fname, "source_row": row, "sha256": h, "caption": r["text"], "blocked": b, "blocked_by": ",".join(block.get(h, []))})
nv = assign_split(rows)
write_manifest(REPO_BENCH / "train" / "multigen_train30" / "manifest.csv", rows, ["sample_id", "source_file", "source_row", "sha256", "caption", "blocked", "blocked_by", "split"])
write_json(REPO_BENCH / "train" / "multigen_train30" / "VERSION.json", {"set": "multigen_train30", "n": len(rows), "n_blocked": nb, "n_val": nv, "val_split": f"500 non-blocked rows, numpy.random.default_rng({VAL_SEED}); use for early stopping / hyperparameters, never the eval sets",
           "source": {"hf_dataset": "limingcv/MultiGen-20M_train", "shards": [Path(f).name for f in files]}, "image": "512x512 PNG bytes in the parquet; canny/depth conditions are extracted at train time with the pinned annotators",
           "rule": "drop rows with blocked = 1 (eval leakage)", "env": env_pins()})
print(f"multigen_train30: {len(rows)} rows, {nb} blocked")

# ------------------------------------------------------------- ade20k_train
files = sorted(glob.glob(str(RAW_ROOT / "captioned_ade20k" / "data" / "train-*.parquet")))
rows, nb = [], 0
for i, (fname, row, r) in enumerate(parquet_rows(files, ["image", "prompt", "image_path"])):
    b = r["image"]["bytes"]; h = sha256_bytes(b)
    w, hh = Image.open(io.BytesIO(b)).size
    _, _, _, box = crop_params(w, hh)
    m = re.search(r"(ADE_train_\d+)", r["image_path"] or "")
    bl = int(h in block); nb += bl
    rows.append({"sample_id": sid(i), "source_file": fname, "source_row": row, "sha256": h, "ade_id": m.group(1) if m else "", "width": w, "height": hh,
                 "crop_box": ",".join(map(str, box)), "caption": r["prompt"], "blocked": bl})
nv = assign_split(rows)
write_manifest(REPO_BENCH / "train" / "ade20k_train" / "manifest.csv", rows, ["sample_id", "source_file", "source_row", "sha256", "ade_id", "width", "height", "crop_box", "caption", "blocked", "split"])
write_json(REPO_BENCH / "train" / "ade20k_train" / "VERSION.json", {"set": "ade20k_train", "n": len(rows), "n_blocked": nb, "n_val": nv, "val_split": f"500 non-blocked rows, numpy.random.default_rng({VAL_SEED})",
           "source": {"hf_dataset": "limingcv/Captioned_ADE20K", "split": "train", "shards": [Path(f).name for f in files]},
           "crop": "same square-crop rule as ade20k_val2k; seg_map cropped with nearest at train time; condition rendered with ade20k_val2k/palette.json",
           "caption": "ControlNet++ 'prompt'", "rule": "drop rows with blocked = 1", "env": env_pins()})
print(f"ade20k_train: {len(rows)} rows, {nb} blocked")

# ------------------------------------------------------------- coco_train
coco = RAW_ROOT / "coco2017"
inst = json.load(open(coco / "annotations" / "instances_train2017.json"))
caps = json.load(open(coco / "annotations" / "captions_train2017.json"))
n_boxes = defaultdict(int)
for a in inst["annotations"]:
    if not a.get("iscrowd", 0):
        n_boxes[a["image_id"]] += 1
cap_by_img = {}
for a in sorted(caps["annotations"], key=lambda a: a["id"]):
    cap_by_img.setdefault(a["image_id"], a)
images = sorted(inst["images"], key=lambda im: im["id"])
rows, nb = [], 0
for i, im in enumerate(images):
    p = coco / "train2017" / im["file_name"]
    h = sha256_file(p)
    _, _, _, box = crop_params(im["width"], im["height"])
    cap = cap_by_img.get(im["id"])
    bl = int(h in block); nb += bl
    rows.append({"sample_id": sid(i), "coco_image_id": im["id"], "file_name": im["file_name"], "sha256": h, "width": im["width"], "height": im["height"],
                 "crop_box": ",".join(map(str, box)), "caption": cap["caption"].strip() if cap else "", "caption_ann_id": cap["id"] if cap else "",
                 "n_boxes_orig": n_boxes.get(im["id"], 0), "blocked": bl})
    if i % 20000 == 0:
        print(f"  coco_train {i}/{len(images)}", flush=True)
nv = assign_split(rows)
write_manifest(REPO_BENCH / "train" / "coco_train" / "manifest.csv", rows, ["sample_id", "coco_image_id", "file_name", "sha256", "width", "height", "crop_box", "caption", "caption_ann_id", "n_boxes_orig", "blocked", "split"])
write_json(REPO_BENCH / "train" / "coco_train" / "VERSION.json", {"set": "coco_train", "n": len(rows), "n_blocked": nb, "n_val": nv, "val_split": f"500 non-blocked rows, numpy.random.default_rng({VAL_SEED})",
           "source": {"images": "COCO 2017 train2017.zip", "annotations": "instances_train2017, captions_train2017"}, "order": "ascending COCO image id",
           "crop": "same square-crop rule as coco_val5k; boxes transformed at train time with common.crop_boxes; condition rendered with coco_val5k/class_colors.json",
           "caption": "lowest caption annotation id", "rule": "drop rows with blocked = 1", "env": env_pins()})
print(f"coco_train: {len(rows)} rows, {nb} blocked")
