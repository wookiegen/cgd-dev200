"""Shared helpers for the CGD benchmark manifests (BENCHMARK_v1.md v1.7, Section 5c).

Conventions (frozen):
- sample_id: five-digit, zero-padded, in SOURCE order (file, then row) so it is reproducible from the source alone.
- sha256: of the ORIGINAL image bytes (parquet bytes or the file on disk), never of a re-encoded copy.
- Square crop: shorter side -> 512 (image bicubic, label map nearest), center crop 512x512; boxes transformed identically,
  dropped when their center falls outside the crop, clipped otherwise. crop_box = (x0, y0, x1, y1) in RESIZED coordinates.
- Files under the data root are rebuilt, never committed; only manifests / JSON pins live in the repo.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 256 * 1024 * 1024   # MultiGen-20M PNGs (and crops saved from them) carry iCCP / text chunks above PIL's 1 MB default

REPO_BENCH = Path(__file__).resolve().parent                     # cgd-dev200/bench  (manifests, pins)
RAW_ROOT = Path(os.environ.get("CGD_RAW_ROOT", "/data/wookiekim/cgd/data"))          # downloaded sources
OUT_ROOT = Path(os.environ.get("CGD_BENCH_ROOT", "/data/wookiekim/cgd/data/bench"))  # materialized sets
SIZE = 512
SUBSET500_SEED = 500


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sid(i: int) -> str:
    return f"{i:05d}"


# ---------------------------------------------------------------- square crop
def crop_params(w: int, h: int, size: int = SIZE):
    """Return (scale, new_w, new_h, crop_box) for shorter-side->size + center crop."""
    s = size / min(w, h)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    x0, y0 = (nw - size) // 2, (nh - size) // 2
    return s, nw, nh, (x0, y0, x0 + size, y0 + size)


def crop_image(img: Image.Image, size: int = SIZE) -> tuple[Image.Image, tuple]:
    s, nw, nh, box = crop_params(*img.size, size)
    if (nw, nh) != img.size:
        img = img.resize((nw, nh), Image.BICUBIC)
    return img.crop(box), box


def crop_label(lab: np.ndarray, size: int = SIZE) -> np.ndarray:
    h, w = lab.shape
    s, nw, nh, (x0, y0, x1, y1) = crop_params(w, h, size)
    im = Image.fromarray(lab)
    if (nw, nh) != (w, h):
        im = im.resize((nw, nh), Image.NEAREST)
    return np.array(im)[y0:y1, x0:x1]


def crop_boxes(boxes_xywh, w: int, h: int, size: int = SIZE):
    """Transform COCO xywh boxes; drop boxes whose center leaves the crop; clip the rest. Returns list of (x, y, bw, bh) in crop coords."""
    s, nw, nh, (x0, y0, x1, y1) = crop_params(w, h, size)
    out = []
    for (x, y, bw, bh) in boxes_xywh:
        X0, Y0, X1, Y1 = x * s - x0, y * s - y0, (x + bw) * s - x0, (y + bh) * s - y0
        cx, cy = (X0 + X1) / 2, (Y0 + Y1) / 2
        if not (0 <= cx < size and 0 <= cy < size):
            out.append(None); continue
        X0, Y0, X1, Y1 = max(0.0, X0), max(0.0, Y0), min(float(size), X1), min(float(size), Y1)
        if X1 - X0 < 1 or Y1 - Y0 < 1:
            out.append(None); continue
        out.append((round(X0, 2), round(Y0, 2), round(X1 - X0, 2), round(Y1 - Y0, 2)))
    return out


# ---------------------------------------------------------------- renderers
def canny_condition(img: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    import cv2
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(gray, low, high)
    return Image.fromarray(np.repeat(e[:, :, None], 3, axis=2))


def render_palette(label: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> Image.Image:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for k, rgb in palette.items():
        lut[int(k)] = rgb
    return Image.fromarray(lut[label])


def render_boxes(boxes, cat_ids, colors: dict[int, tuple[int, int, int]], size: int = SIZE) -> Image.Image:
    """Filled class-color boxes on black, no text. Larger boxes first so small ones stay visible."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    order = sorted(range(len(boxes)), key=lambda i: -(boxes[i][2] * boxes[i][3]))
    for i in order:
        x, y, bw, bh = boxes[i]
        X0, Y0, X1, Y1 = int(round(x)), int(round(y)), int(round(x + bw)), int(round(y + bh))
        canvas[Y0:max(Y1, Y0 + 1), X0:max(X1, X0 + 1)] = colors[int(cat_ids[i])]
    return Image.fromarray(canvas)


def distinct_colors(n: int, seed: int = 0) -> list[tuple[int, int, int]]:
    """Deterministic, well-separated RGB colors (golden-ratio hue walk, two S/V rings), for the class-color maps."""
    import colorsys
    cols = []
    h = 0.0
    for i in range(n):
        h = (h + 0.61803398875) % 1.0
        s, v = (0.95, 0.95) if i % 2 == 0 else (0.65, 0.85)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return cols


# ---------------------------------------------------------------- manifests / pins
def write_manifest(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def env_pins() -> dict:
    pins = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ["numpy", "PIL", "cv2", "torch", "transformers", "pyarrow"]:
        try:
            m = __import__(mod)
            pins[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
    return pins


def subset500_flags(n: int) -> np.ndarray:
    flags = np.zeros(n, dtype=bool)
    if n >= 500:
        flags[np.random.default_rng(SUBSET500_SEED).choice(n, 500, replace=False)] = True
    return flags


def load_dev200_shas() -> dict[str, int]:
    p = REPO_BENCH.parent / "dev200" / "manifest.csv"
    if not p.exists():
        return {}
    with open(p) as f:
        return {r["sha256"]: int(r["dev_idx"]) for r in csv.DictReader(f)}


def parquet_rows(files, columns):
    """Yield (file_name, row_in_file, dict) over parquet shards in the given order, streaming by batch."""
    import pyarrow.parquet as pq
    for fp in files:
        pf = pq.ParquetFile(fp)
        row = 0
        for batch in pf.iter_batches(batch_size=64, columns=columns):
            d = batch.to_pydict()
            n = len(d[columns[0]])
            for i in range(n):
                yield Path(fp).name, row, {c: d[c][i] for c in columns}
                row += 1
