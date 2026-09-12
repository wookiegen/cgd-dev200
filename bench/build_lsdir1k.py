"""Build the lsdir1k validation split (BENCHMARK v1.13): REAL high-resolution photographs for validating the c2048 column.

Source: LSDIR (Li et al., CVPRW 2023; 84,991 training HR images), read from the Hugging Face parquet mirror danjacobellis/LSDIR, whose
rows carry the source path and the image size, so the selection needs no image bytes. Rows with min(w, h) >= --min-side (2048) are
shuffled with a fixed seed and the first N (1000) are materialized:
  <OUT>/lsdir1k/images2048/<sid>.png          center crop of the ORIGINAL pixels, 2048 x 2048, no resampling   (the 2048 reference; real)
  <OUT>/lsdir1k/images/<sid>.png              cv2.INTER_AREA downsample of that crop to 512                     (the 512 real image the pipeline sees)
  <OUT>/lsdir1k/conditions/canny/<sid>.png    c512  = cv2.Canny(gray(512 image), 100, 200), 3-channel           (the controller's input, as in multigen5k)
  <OUT>/lsdir1k/conditions/canny2048/<sid>.png c2048 = cv2.Canny(gray(2048 real crop), 100, 200), 3-channel    (the edge map of the REAL 2048 image)
  bench/lsdir1k/manifest.csv                  sample_id, source, sizes, crop offset, sha256 of the source bytes, caption
So on this split the real image, not the PiD round trip, is the reference of the c2048 column at 1.0, and the round trip becomes a row.

Stages (each resumable; run inside the container):
  python build_lsdir1k.py --scan                 metadata only (path, w, h of every row; ~15 min of range reads) -> bench/lsdir1k/sizes.json
  python build_lsdir1k.py --build [--n 1000]     stream the needed row groups, crop, write conditions + manifest (CPU; network)
  CUDA_VISIBLE_DEVICES=g python build_lsdir1k.py --captions   one-sentence captions with Qwen2.5-VL-7B-Instruct (cached in HF_HOME) -> manifest
"""
import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 256 * 1024 * 1024

from common import OUT_ROOT, REPO_BENCH, env_pins, sid, write_json, write_manifest  # noqa: E402

REPO = "danjacobellis/LSDIR"
SPLIT = "lsdir1k"
SIZES = REPO_BENCH / SPLIT / "sizes.json"
MANIFEST = REPO_BENCH / SPLIT / "manifest.csv"
COLUMNS = ["sample_id", "source_file", "source_row_group", "source_index", "lsdir_path", "sha256", "source_format", "width", "height",
           "crop_x0", "crop_y0", "caption", "dup_group", "dev200_idx", "subset500"]
CROP = 2048
CAPTION_PROMPT = ("Write one sentence that describes this photograph as a caption for an image generator: the main subject, the setting, "
                  "and the lighting or style. Plain text, no quotes, do not start with 'The image' or 'This image'.")

ap = argparse.ArgumentParser()
ap.add_argument("--scan", action="store_true"); ap.add_argument("--build", action="store_true"); ap.add_argument("--captions", action="store_true")
ap.add_argument("--n", type=int, default=1000); ap.add_argument("--min-side", type=int, default=CROP); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--captioner", default="Qwen/Qwen2.5-VL-7B-Instruct"); ap.add_argument("--caption-batch", type=int, default=8)
a = ap.parse_args()
(REPO_BENCH / SPLIT).mkdir(parents=True, exist_ok=True)
out = OUT_ROOT / SPLIT
for d in ["images2048", "images", "conditions/canny", "conditions/canny2048"]:
    (out / d).mkdir(parents=True, exist_ok=True)


def hf_fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def canny3(rgb: np.ndarray) -> np.ndarray:
    e = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 100, 200)
    return np.repeat(e[:, :, None], 3, axis=2)


# ---------------------------------------------------------------- 1. scan: sizes of every row from the parquet metadata columns
if a.scan and not SIZES.exists():
    import pyarrow.parquet as pq
    fs = hf_fs()
    files = sorted(fs.glob(f"datasets/{REPO}/data/*.parquet"))
    rows = []; t0 = time.time()
    for i, f in enumerate(files):
        with fs.open(f, "rb") as fh:
            pf = pq.ParquetFile(fh)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=["path", "w", "h"]).to_pydict()
                rows += [{"file": f.split("/")[-1], "rg": rg, "idx": j, "path": p, "w": w, "h": h} for j, (p, w, h) in enumerate(zip(t["path"], t["w"], t["h"]))]
        if i % 20 == 0:
            print(f"  scan {i}/{len(files)} shards, {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)
    json.dump(rows, open(SIZES, "w"))
    ms = np.array([min(r["w"], r["h"]) for r in rows])
    print(f"scanned {len(rows)} rows; min side >= 2048: {(ms >= 2048).sum()}, >= 1536: {(ms >= 1536).sum()}; median min side {np.median(ms):.0f}")

# ---------------------------------------------------------------- 2. build: select, stream row groups, crop, conditions, manifest
if a.build:
    import pyarrow.parquet as pq
    if not SIZES.exists():
        raise SystemExit("run --scan first")
    rows = json.load(open(SIZES))
    ok = [r for r in rows if min(r["w"], r["h"]) >= a.min_side]
    # row groups are the unit of download (one may be a whole ~460 MB shard), so the sample is drawn group by group: shuffle the groups
    # with the seed and take every qualifying row of a group until N are selected; groups partition LSDIR arbitrarily (by upload order)
    groups = {}
    for r in ok:
        groups.setdefault((r["file"], r["rg"]), []).append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(a.seed)
    sel = []
    for gi in rng.permutation(len(keys)):
        sel += groups[keys[gi]]
        if len(sel) >= a.n:
            break
    sel = sel[: a.n]
    print(f"{len(rows)} rows scanned; {len(ok)} with min side >= {a.min_side} in {len(keys)} row groups; selecting {len(sel)} from "
          f"{len({(r['file'], r['rg']) for r in sel})} groups (seed {a.seed})", flush=True)
    # sample ids follow the selection order; the manifest is rebuilt from what exists so the stage is resumable
    for k, r in enumerate(sel):
        r["sample_id"] = sid(k)
    existing = {}
    if MANIFEST.exists():
        existing = {r["sample_id"]: r for r in csv.DictReader(open(MANIFEST))}
    fs = hf_fs()
    by_group = {}
    for r in sel:
        if not (out / "images2048" / f"{r['sample_id']}.png").exists():
            by_group.setdefault((r["file"], r["rg"]), []).append(r)
    print(f"{sum(len(v) for v in by_group.values())} images to materialize from {len(by_group)} row groups", flush=True)
    t0 = time.time(); n = 0
    manifest = dict(existing)
    for (f, rg), items in sorted(by_group.items()):
        with fs.open(f"datasets/{REPO}/data/{f}", "rb") as fh:
            pf = pq.ParquetFile(fh)
            t = pf.read_row_group(rg, columns=["path", "image", "w", "h"]).to_pydict()
        for r in items:
            j = r["idx"]
            assert t["path"][j] == r["path"], (f, rg, j, t["path"][j], r["path"])
            b = t["image"][j]["bytes"]
            img = Image.open(io.BytesIO(b)); fmt = img.format or ""
            img = img.convert("RGB"); W, H = img.size
            assert min(W, H) >= a.min_side, (r["path"], W, H)
            x0 = (W - CROP) // 2; y0 = (H - CROP) // 2
            crop = np.array(img.crop((x0, y0, x0 + CROP, y0 + CROP)))
            view = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
            s = r["sample_id"]
            cv2.imwrite(str(out / "conditions" / "canny2048" / f"{s}.png"), canny3(crop))
            cv2.imwrite(str(out / "conditions" / "canny" / f"{s}.png"), canny3(view))
            cv2.imwrite(str(out / "images" / f"{s}.png"), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / "images2048" / f"{s}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))   # last: its existence marks the row done
            manifest[s] = {"sample_id": s, "source_file": f, "source_row_group": rg, "source_index": j, "lsdir_path": r["path"],
                           "sha256": hashlib.sha256(b).hexdigest(), "source_format": fmt, "width": W, "height": H, "crop_x0": x0, "crop_y0": y0,
                           "caption": existing.get(s, {}).get("caption", ""), "dup_group": "", "dev200_idx": "", "subset500": ""}
            n += 1
            if n % 50 == 0:
                print(f"  [{n}] {time.time()-t0:.0f}s ({(time.time()-t0)/n:.1f}s/img)", flush=True)
                write_manifest(MANIFEST, [manifest[k] for k in sorted(manifest)], COLUMNS)
    for r in sel:   # rows already on disk but missing from the manifest (interrupted run) are re-described without re-downloading the bytes
        s = r["sample_id"]
        if s not in manifest and (out / "images2048" / f"{s}.png").exists():
            manifest[s] = {"sample_id": s, "source_file": r["file"], "source_row_group": r["rg"], "source_index": r["idx"], "lsdir_path": r["path"],
                           "sha256": "", "source_format": "", "width": r["w"], "height": r["h"], "crop_x0": (r["w"] - CROP) // 2, "crop_y0": (r["h"] - CROP) // 2,
                           "caption": "", "dup_group": "", "dev200_idx": "", "subset500": ""}
    write_manifest(MANIFEST, [manifest[k] for k in sorted(manifest)], COLUMNS)
    write_json(REPO_BENCH / SPLIT / "VERSION.json", {
        "split": SPLIT, "benchmark_version": "v1.13", "source": f"hf://datasets/{REPO} (parquet mirror of LSDIR; Li et al., CVPRW 2023 'LSDIR: A Large Scale Dataset for Image Restoration')",
        "selection": f"rows with min(w, h) >= {a.min_side}, seeded shuffle (numpy default_rng({a.seed})), first {a.n}", "n": len(manifest),
        "images2048": "center crop of the original pixels, 2048 x 2048, no resampling (the real 2048 reference)",
        "images": "cv2.INTER_AREA downsample of the crop to 512 (the real 512 image)",
        "conditions/canny": "c512 = cv2.Canny(gray(512 image), 100, 200)", "conditions/canny2048": "c2048 = cv2.Canny(gray(2048 crop), 100, 200)",
        "reference_of_c2048_column": "the real 2048 image (1.0); the PiD round trip is a row here, not the reference",
        "captions": "one sentence per image by the captioner recorded in CAPTIONS.json (stage --captions)", "env": env_pins()})
    print(f"lsdir1k: {len(manifest)} rows materialized in {time.time()-t0:.0f}s -> {out}; manifest {MANIFEST}")

# ---------------------------------------------------------------- 3. captions: Qwen2.5-VL, one sentence per 512 image
if a.captions:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    rows = list(csv.DictReader(open(MANIFEST)))
    todo = [r for r in rows if not r["caption"].strip()]
    print(f"captions: {len(todo)} of {len(rows)} rows need one ({a.captioner})", flush=True)
    if todo:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.captioner, torch_dtype=torch.bfloat16, device_map="cuda")
        proc = AutoProcessor.from_pretrained(a.captioner, min_pixels=256 * 28 * 28, max_pixels=512 * 512)
        proc.tokenizer.padding_side = "left"
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": CAPTION_PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t0 = time.time()
        for i in range(0, len(todo), a.caption_batch):
            batch = todo[i: i + a.caption_batch]
            imgs = [Image.open(out / "images" / f"{r['sample_id']}.png").convert("RGB") for r in batch]
            inputs = proc(text=[text] * len(batch), images=imgs, padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            outs = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, o in zip(batch, outs):
                c = o.strip().splitlines()[0].strip().strip('"').strip()
                r["caption"] = c[:300]
            if (i // a.caption_batch) % 10 == 0:
                print(f"  [{i+len(batch)}/{len(todo)}] {time.time()-t0:.0f}s  e.g. {batch[0]['sample_id']}: {batch[0]['caption']}", flush=True)
                write_manifest(MANIFEST, rows, COLUMNS)
        write_manifest(MANIFEST, rows, COLUMNS)
        write_json(REPO_BENCH / SPLIT / "CAPTIONS.json", {"captioner": a.captioner, "prompt": CAPTION_PROMPT, "input": "the 512 image", "decoding": "greedy, max 64 new tokens",
                                                          "n": len(rows), "env": env_pins()})
    print(f"captions done: {sum(1 for r in rows if r['caption'].strip())}/{len(rows)}")
