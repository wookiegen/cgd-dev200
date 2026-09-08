#!/usr/bin/env python
"""Select the FIXED dev-200 subset of the MultiGen-20M canny eval split and materialize it.

Source: HF dataset limingcv/MultiGen-20M_canny_eval, the 5 validation parquet shards
(5000 images, 512x512, with captions). This is the 5k evaluation set used by ControlNet++.
Selection: numpy default_rng(seed=0).choice(5000, 200, replace=False), over the shards
concatenated in filename order. The result is frozen in dev200/manifest.csv; never regenerate
with a different seed under the same name.

Outputs (under --out):
  manifest.csv                dev_idx, val_row, sha256(image bytes), caption
  images/{dev_idx:03d}.png    the 512x512 source image (paired reference for LPIPS/PSNR)
  canny/{dev_idx:03d}.png     the pinned condition: cv2.Canny(gray, 100, 200), replicated to RGB
  captions.json               dev_idx -> caption (the prompt)
"""
import argparse, csv, glob, hashlib, io, json, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CGD_ROOT = os.environ.get("CGD_ROOT", os.path.dirname(REPO))   # parent dir holding cgd-dev200/, OminiControl/, PiD/, data/
import numpy as np, cv2, pyarrow.parquet as pq
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("MULTIGEN_DIR", os.path.join(CGD_ROOT, "data", "multigen_canny_eval", "data")))
ap.add_argument("--out", default=os.path.join(REPO, "dev200"))
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--canny-low", type=int, default=100)
ap.add_argument("--canny-high", type=int, default=200)
a = ap.parse_args()

shards = sorted(glob.glob(os.path.join(a.data, "validation-*.parquet")))
assert len(shards) == 5, shards
rows = []
for s in shards:
    t = pq.read_table(s, columns=["image", "text"])
    rows.extend(t.to_pylist())
assert len(rows) == 5000, len(rows)

rng = np.random.default_rng(a.seed)
pick = sorted(rng.choice(len(rows), a.n, replace=False).tolist())

os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
os.makedirs(os.path.join(a.out, "canny"), exist_ok=True)
caps = {}
with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dev_idx", "val_row", "sha256", "caption"])
    for i, r in enumerate(pick):
        b = rows[r]["image"]["bytes"]; cap = rows[r]["text"]
        im = Image.open(io.BytesIO(b)).convert("RGB")
        assert im.size == (512, 512), im.size
        im.save(os.path.join(a.out, "images", f"{i:03d}.png"))
        g = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(g, a.canny_low, a.canny_high)
        Image.fromarray(np.stack([e] * 3, -1)).save(os.path.join(a.out, "canny", f"{i:03d}.png"))
        w.writerow([i, r, hashlib.sha256(b).hexdigest(), cap]); caps[f"{i:03d}"] = cap
json.dump(caps, open(os.path.join(a.out, "captions.json"), "w"), indent=1, ensure_ascii=False)
print(f"selected {len(pick)} of {len(rows)} (seed {a.seed}); first val_rows: {pick[:8]}")
print(f"wrote {a.out}/manifest.csv, images/, canny/, captions.json")
