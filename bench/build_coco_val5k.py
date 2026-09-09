"""coco_val5k: COCO 2017 val (5000 images) with GT boxes + class labels and one pinned caption per image.

Source: /data/.../coco2017/{val2017, annotations/instances_val2017.json, annotations/captions_val2017.json}. Order = ascending COCO image id.
Square crop per BENCHMARK v1.7; boxes transformed identically (center outside crop -> dropped, else clipped). iscrowd boxes dropped.
Caption = the caption annotation with the LOWEST id for that image. Condition = filled class-color boxes on black (class_colors.json), larger first.
Writes: images/<id>.png (cropped), conditions/bbox/<id>.png, boxes.jsonl (repo), manifest.csv, class_colors.json, VERSION.json.
Layout metrics run on samples with n_boxes_after_crop >= 1 (has_boxes = 1); FID/pFID run on all 5000.
"""
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from common import (OUT_ROOT, RAW_ROOT, REPO_BENCH, crop_boxes, crop_image, distinct_colors, env_pins, render_boxes, sha256_file, sid,
                    subset500_flags, write_json, write_manifest)

SET = "coco_val5k"
out = OUT_ROOT / SET
for d in ["images", "conditions/bbox"]:
    (out / d).mkdir(parents=True, exist_ok=True)
coco = RAW_ROOT / "coco2017"
inst = json.load(open(coco / "annotations" / "instances_val2017.json"))
caps = json.load(open(coco / "annotations" / "captions_val2017.json"))
cats = sorted(inst["categories"], key=lambda c: c["id"])
colors_list = distinct_colors(len(cats))
colors = {c["id"]: colors_list[k] for k, c in enumerate(cats)}
anns = defaultdict(list)
for a in inst["annotations"]:
    if not a.get("iscrowd", 0):
        anns[a["image_id"]].append(a)
cap_by_img = {}
for a in sorted(caps["annotations"], key=lambda a: a["id"]):
    cap_by_img.setdefault(a["image_id"], a)
images = sorted(inst["images"], key=lambda im: im["id"])
assert len(images) == 5000
rows, box_lines = [], []
n_dropped = 0
for i, im in enumerate(images):
    s = sid(i)
    p = coco / "val2017" / im["file_name"]
    img = Image.open(p).convert("RGB")
    w, h = img.size
    assert (w, h) == (im["width"], im["height"]), (im["file_name"], img.size)
    cimg, box = crop_image(img)
    cimg.save(out / "images" / f"{s}.png")
    a_list = sorted(anns.get(im["id"], []), key=lambda a: a["id"])
    tb = crop_boxes([a["bbox"] for a in a_list], w, h)
    kept = [(a, b) for a, b in zip(a_list, tb) if b is not None]
    n_dropped += len(a_list) - len(kept)
    render_boxes([b for _, b in kept], [a["category_id"] for a, _ in kept], colors).save(out / "conditions" / "bbox" / f"{s}.png")
    cap = cap_by_img.get(im["id"])
    rows.append({"sample_id": s, "coco_image_id": im["id"], "file_name": im["file_name"], "sha256": sha256_file(p), "width": w, "height": h,
                 "crop_box": ",".join(map(str, box)), "caption": cap["caption"].strip() if cap else "", "caption_ann_id": cap["id"] if cap else "",
                 "n_boxes_orig": len(a_list), "n_boxes_after_crop": len(kept), "has_boxes": int(len(kept) > 0)})
    box_lines.append({"sample_id": s, "coco_image_id": im["id"], "boxes": [
        {"ann_id": a["id"], "category_id": a["category_id"], "name": next(c["name"] for c in cats if c["id"] == a["category_id"]),
         "bbox_xywh_crop": list(b), "bbox_xywh_orig": [round(v, 2) for v in a["bbox"]]} for a, b in kept]})
flags = subset500_flags(len(rows))
for j, r in enumerate(rows):
    r["subset500"] = int(flags[j])
cols = ["sample_id", "coco_image_id", "file_name", "sha256", "width", "height", "crop_box", "caption", "caption_ann_id", "n_boxes_orig",
        "n_boxes_after_crop", "has_boxes", "subset500"]
write_manifest(REPO_BENCH / SET / "manifest.csv", rows, cols)
(REPO_BENCH / SET).mkdir(parents=True, exist_ok=True)
with open(REPO_BENCH / SET / "boxes.jsonl", "w") as f:
    for line in box_lines:
        f.write(json.dumps(line) + "\n")
write_json(REPO_BENCH / SET / "class_colors.json", {"note": "COCO category_id -> RGB used to render the bbox condition (filled boxes on black, larger boxes drawn first)",
                                                    "classes": [{"category_id": c["id"], "name": c["name"], "rgb": list(colors[c["id"]])} for c in cats]})
write_json(REPO_BENCH / SET / "VERSION.json", {
    "set": SET, "n": len(rows), "n_with_boxes": int(sum(r["has_boxes"] for r in rows)), "n_boxes_dropped_by_crop": n_dropped,
    "source": {"images": "COCO 2017 val2017.zip", "annotations": "annotations_trainval2017.zip (instances_val2017, captions_val2017)", "url": "https://cocodataset.org"},
    "order": "ascending COCO image id", "crop": "shorter side -> 512 bicubic, center crop 512x512; boxes: center outside crop -> dropped, else clipped (BENCHMARK v1.7)",
    "boxes": "iscrowd dropped; boxes.jsonl lists kept boxes in crop coords and original coords", "caption": "lowest caption annotation id per image",
    "conditions": {"bbox": "class_colors.json filled boxes on black, no text, larger first"}, "scorer": "IDEA-Research/grounding-dino-base, MIGC protocol (SR @ IoU 0.5, mIoU)",
    "subset500": "numpy.random.default_rng(500).choice(n, 500, replace=False)", "env": env_pins()})
print(f"{SET}: {len(rows)} rows, {sum(r['has_boxes'] for r in rows)} with boxes after crop, {n_dropped} boxes dropped by crop")
