"""ade20k_val2k: ADE20K validation (2000 images) with GT semantic masks and the ControlNet++ captions.

Source: HF limingcv/Captioned_ADE20K, validation shards in filename order. seg_map: uint8, 0 = other/ignore, 1..150 = ADE20K classes
(Mask2Former ADE semantic predicts 0..149 = label - 1). Square crop per BENCHMARK v1.7 (shorter side 512 bicubic / nearest, center crop).
Writes: images/<id>.png (cropped), labels/<id>.png (cropped uint8 label map), conditions/seg/<id>.png (palette render), manifest.csv,
palette.json (derived from the dataset's own control_seg so our render matches the ControlNet++ convention), VERSION.json.
"""
import glob
import io
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from common import (OUT_ROOT, RAW_ROOT, REPO_BENCH, crop_image, crop_label, env_pins, parquet_rows, render_palette, sha256_bytes, sid,
                    subset500_flags, write_json, write_manifest)

SET = "ade20k_val2k"
out = OUT_ROOT / SET
for d in ["images", "labels", "conditions/seg"]:
    (out / d).mkdir(parents=True, exist_ok=True)

files = sorted(glob.glob(str(RAW_ROOT / "captioned_ade20k" / "data" / "validation-*.parquet")))
assert len(files) == 3, files

# pass 1: derive the palette (label -> dominant control_seg color) and build rows
color_votes: dict[int, Counter] = defaultdict(Counter)
rows, pending = [], []
i = 0
for fname, row, r in parquet_rows(files, ["image", "prompt", "control_seg", "seg_map", "image_path"]):
    b = r["image"]["bytes"]
    img = Image.open(io.BytesIO(b)).convert("RGB")
    seg = np.array(r["seg_map"], dtype=np.uint8)
    assert seg.shape == (img.size[1], img.size[0]), (fname, row, seg.shape, img.size)
    cs = np.array(Image.open(io.BytesIO(r["control_seg"]["bytes"])).convert("RGB"))
    for lab in np.unique(seg):
        px = cs[seg == lab]
        if len(px) > 50:
            u, c = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
            color_votes[int(lab)][tuple(int(v) for v in u[c.argmax()])] += int(c.max())
    s = sid(i)
    w, h = img.size
    cimg, box = crop_image(img)
    clab = crop_label(seg)
    cimg.save(out / "images" / f"{s}.png")
    Image.fromarray(clab).save(out / "labels" / f"{s}.png")
    m = re.search(r"(ADE_val_\d+)", r["image_path"] or "")
    rows.append({"sample_id": s, "source_file": fname, "source_row": row, "sha256": sha256_bytes(b), "ade_id": m.group(1) if m else "",
                 "width": w, "height": h, "crop_box": ",".join(map(str, box)), "caption": r["prompt"], "caption_source": "ControlNet++ Captioned_ADE20K 'prompt'",
                 "n_labels_after_crop": int(len(np.unique(clab[clab > 0])))})
    pending.append((s, clab))
    i += 1
assert i == 2000, i
palette = {lab: max(v.items(), key=lambda kv: kv[1])[0] for lab, v in color_votes.items()}
palette[0] = (0, 0, 0)
ambiguous = {lab: v.most_common(2) for lab, v in color_votes.items() if len(v) > 1 and v.most_common(2)[1][1] > 0.05 * sum(v.values())}
# pass 2: render our condition with the derived palette
for s, clab in pending:
    render_palette(clab, palette).save(out / "conditions" / "seg" / f"{s}.png")
flags = subset500_flags(len(rows))
for j, r in enumerate(rows):
    r["subset500"] = int(flags[j])
cols = ["sample_id", "source_file", "source_row", "sha256", "ade_id", "width", "height", "crop_box", "caption", "caption_source", "n_labels_after_crop", "subset500"]
write_manifest(REPO_BENCH / SET / "manifest.csv", rows, cols)
write_json(REPO_BENCH / SET / "palette.json", {"note": "label id (1..150, 0 = other/ignore) -> RGB, derived from Captioned_ADE20K control_seg; matches the standard ADE20K palette with label-1 indexing",
                                                "labels_seen": len(palette) - 1, "ambiguous_labels": {str(k): v for k, v in ambiguous.items()},
                                                "palette": {str(k): list(v) for k, v in sorted(palette.items())}})
write_json(REPO_BENCH / SET / "VERSION.json", {
    "set": SET, "n": len(rows), "source": {"hf_dataset": "limingcv/Captioned_ADE20K", "split": "validation", "shards": [Path(f).name for f in files]},
    "crop": "shorter side -> 512 (image bicubic, label nearest), center crop 512x512; crop_box = x0,y0,x1,y1 in resized coords (BENCHMARK v1.7)",
    "labels": "uint8 PNG, 0 = other/ignore (excluded from mIoU), 1..150 = ADE20K classes; Mask2Former ADE semantic id = label - 1",
    "conditions": {"seg": "palette.json render of the cropped label map, 0 -> black"},
    "caption": "ControlNet++ 'prompt' column (BENCHMARK v1.6)", "scorer": "facebook/mask2former-swin-large-ade-semantic",
    "subset500": "numpy.random.default_rng(500).choice(n, 500, replace=False)", "env": env_pins()})
print(f"{SET}: {len(rows)} rows; palette labels seen {len(palette)-1}/150; ambiguous {len(ambiguous)}; crop boxes e.g. {rows[0]['crop_box']}")
