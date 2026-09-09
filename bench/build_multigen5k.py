"""multigen5k: the MultiGen-20M evaluation split (5000 validation rows) shared by the canny and depth conditions.

Source: HF limingcv/MultiGen-20M_canny_eval, validation shards in filename order (verified identical to MultiGen-20M_depth_eval,
same images, same order, 2026-09-09). Images are 512x512 PNG; all 5000 rows are kept (25 byte-identical duplicates -> dup_group).
Writes: images/<id>.png (original bytes), conditions/canny/<id>.png, manifest.csv, VERSION.json. Depth conditions: depth_dpt.py.
"""
import glob
import io
from collections import defaultdict
from pathlib import Path

from PIL import Image

from common import (OUT_ROOT, RAW_ROOT, REPO_BENCH, canny_condition, env_pins, load_dev200_shas, parquet_rows, sha256_bytes, sid,
                    subset500_flags, write_json, write_manifest)

SET = "multigen5k"
out = OUT_ROOT / SET
(out / "images").mkdir(parents=True, exist_ok=True)
(out / "conditions" / "canny").mkdir(parents=True, exist_ok=True)

files = sorted(glob.glob(str(RAW_ROOT / "multigen_canny_eval" / "data" / "validation-*.parquet")))
assert len(files) == 5, files
dev200 = load_dev200_shas()
rows, by_sha = [], defaultdict(list)
formats_reencoded = defaultdict(int)
i = 0
for fname, row, r in parquet_rows(files, ["image", "text"]):
    b = r["image"]["bytes"]
    h = sha256_bytes(b)
    img = Image.open(io.BytesIO(b)); img.load()
    fmt = img.format or "?"
    assert img.size == (512, 512), (fname, row, img.size, fmt)
    s = sid(i)
    if fmt == "PNG":
        with open(out / "images" / f"{s}.png", "wb") as f:
            f.write(b)                       # original bytes, sha256 verifies the file itself
    else:
        img.convert("RGB").save(out / "images" / f"{s}.png", compress_level=1)   # lossless re-encode (a few rows are TIFF); sha256 is of the SOURCE bytes
        formats_reencoded[fmt] += 1
    canny_condition(img.convert("RGB")).save(out / "conditions" / "canny" / f"{s}.png")
    rows.append({"sample_id": s, "source_file": fname, "source_row": row, "sha256": h, "source_format": fmt, "width": 512, "height": 512,
                 "caption": r["text"], "dev200_idx": dev200.get(h, ""), "dup_group": ""})
    by_sha[h].append(i)
    i += 1
assert i == 5000, i
g = 0
for h, idx in by_sha.items():
    if len(idx) > 1:
        for j in idx:
            rows[j]["dup_group"] = f"d{g:02d}"
        g += 1
flags = subset500_flags(len(rows))
for j, r in enumerate(rows):
    r["subset500"] = int(flags[j])
cols = ["sample_id", "source_file", "source_row", "sha256", "source_format", "width", "height", "caption", "dup_group", "dev200_idx", "subset500"]
write_manifest(REPO_BENCH / SET / "manifest.csv", rows, cols)
write_json(REPO_BENCH / SET / "VERSION.json", {
    "set": SET, "n": len(rows), "n_unique_images": len(by_sha), "n_dup_groups": g,
    "source": {"hf_dataset": "limingcv/MultiGen-20M_canny_eval", "split": "validation", "shards": [Path(f).name for f in files],
               "note": "identical images/order to limingcv/MultiGen-20M_depth_eval validation (sha256 check 2026-09-09)"},
    "image": "512x512, no crop; PNG sources are written as their original bytes, other formats (see source_format, e.g. TIFF) are losslessly re-encoded to PNG; sha256 is always of the SOURCE bytes",
    "n_reencoded_by_format": dict(formats_reencoded),
    "conditions": {"canny": {"annotator": "cv2.Canny(grayscale, 100, 200)", "format": "3-channel PNG, edges = 255", "grayscale": "cv2.COLOR_RGB2GRAY"},
                   "depth": "see depth_dpt.py / DEPTH_VERSION.json"},
    "caption": "dataset 'text' column", "dev200_idx": "row index in dev200/manifest.csv when the image is in the dev-200 set (matched by sha256)",
    "subset500": f"numpy.random.default_rng({500}).choice(n, 500, replace=False)", "env": env_pins()})
print(f"{SET}: {len(rows)} rows, {len(by_sha)} unique images, {g} dup groups, dev200 matched {sum(1 for r in rows if r['dev200_idx'] != '')}")
