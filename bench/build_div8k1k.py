"""Build the div8k1k validation split (BENCHMARK v1.13): REAL high-resolution photographs for validating the c2048 column.

Source: DIV8K (Gu et al., ICCVW 2019; 1500 photographs with resolutions from 2K to 8K), the zip on the Hugging Face Hub
(Iceclear/DIV8K_TrainingSet, DIV8K.zip, 43 GB, read from the HF cache). LSDIR, the first choice, has no images at 2048 on the short side
(its HR images are ~1K; checked 2026-09-12), which is why this split uses DIV8K. Members whose short side is >= 2048 are shuffled with a
fixed seed and the first N (1000) are materialized:
  <OUT>/div8k1k/images2048/<sid>.png          area-downsampled so that the short side is 2048, then the 2048 x 2048 center crop
                                              (a real photograph at 2048, never upsampled: the 2048 reference, the 1.0 of the c2048 column)
  <OUT>/div8k1k/images/<sid>.png              cv2.INTER_AREA downsample of that crop to 512                 (the real 512 image the pipeline sees)
  <OUT>/div8k1k/conditions/canny/<sid>.png    c512  = cv2.Canny(gray(512 image), 100, 200), 3-channel       (the controller's input, as in multigen5k)
  <OUT>/div8k1k/conditions/canny2048/<sid>.png c2048 = cv2.Canny(gray(2048 crop), 100, 200), 3-channel     (the edge map of the REAL 2048 image)
  bench/div8k1k/manifest.csv                  sample_id, zip member, sizes, resize + crop offsets, sha256 of the member bytes, caption

Stages (each resumable; run inside the container):
  python build_div8k1k.py --build [--n 1000]     scan member sizes (cached in bench/div8k1k/sizes.json), select, crop, conditions, manifest (CPU)
  CUDA_VISIBLE_DEVICES=g python build_div8k1k.py --captions   one-sentence captions with Qwen2.5-VL-7B-Instruct (cached in HF_HOME) -> manifest
"""
import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 256 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = None

from common import OUT_ROOT, REPO_BENCH, env_pins, sid, write_json, write_manifest  # noqa: E402

REPO = "Iceclear/DIV8K_TrainingSet"; ZIP = "DIV8K.zip"
SPLIT = "div8k1k"
SIZES = REPO_BENCH / SPLIT / "sizes.json"
MANIFEST = REPO_BENCH / SPLIT / "manifest.csv"
COLUMNS = ["sample_id", "source_file", "sha256", "source_format", "width", "height", "resized_width", "resized_height", "crop_x0", "crop_y0",
           "caption", "dup_group", "dev200_idx", "subset500"]
CROP = 2048
CAPTION_PROMPT = ("Write one sentence that describes this photograph as a caption for an image generator: the main subject, the setting, "
                  "and the lighting or style. Plain text, no quotes, do not start with 'The image' or 'This image'.")

ap = argparse.ArgumentParser()
ap.add_argument("--build", action="store_true"); ap.add_argument("--captions", action="store_true")
ap.add_argument("--n", type=int, default=1000); ap.add_argument("--min-side", type=int, default=CROP); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--zip", default=None, help="path of DIV8K.zip (default: the HF cache copy of Iceclear/DIV8K_TrainingSet)")
ap.add_argument("--captioner", default="Qwen/Qwen2.5-VL-7B-Instruct"); ap.add_argument("--caption-batch", type=int, default=8)
a = ap.parse_args()
(REPO_BENCH / SPLIT).mkdir(parents=True, exist_ok=True)
out = OUT_ROOT / SPLIT
for d in ["images2048", "images", "conditions/canny", "conditions/canny2048"]:
    (out / d).mkdir(parents=True, exist_ok=True)


def canny3(rgb: np.ndarray) -> np.ndarray:
    e = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 100, 200)
    return np.repeat(e[:, :, None], 3, axis=2)


def zip_path() -> str:
    if a.zip:
        return a.zip
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, ZIP, repo_type="dataset")


# ---------------------------------------------------------------- 1. build: sizes, select, crop, conditions, manifest
if a.build:
    zp = zip_path(); print("zip:", zp, flush=True)
    z = zipfile.ZipFile(zp)
    members = sorted(n for n in z.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")))
    if SIZES.exists():
        sizes = json.load(open(SIZES))
    else:
        sizes = {}; t0 = time.time()
        for k, n in enumerate(members):
            with z.open(n) as fh:
                w, h = Image.open(fh).size          # header only
            sizes[n] = [w, h]
            if (k + 1) % 200 == 0:
                print(f"  sizes {k+1}/{len(members)} {time.time()-t0:.0f}s", flush=True)
        json.dump(sizes, open(SIZES, "w"))
    ms = np.array([min(v) for v in sizes.values()])
    print(f"{len(members)} members; min side median {np.median(ms):.0f}, >= {a.min_side}: {(ms >= a.min_side).sum()}", flush=True)
    ok = [n for n in members if min(sizes[n]) >= a.min_side]
    rng = np.random.default_rng(a.seed)
    sel = [ok[i] for i in rng.permutation(len(ok))[: a.n]]
    print(f"selecting {len(sel)} (seed {a.seed})", flush=True)
    existing = {r["sample_id"]: r for r in csv.DictReader(open(MANIFEST))} if MANIFEST.exists() else {}
    manifest = dict(existing); t0 = time.time(); n_new = 0
    for k, name in enumerate(sel):
        s = sid(k)
        if (out / "images2048" / f"{s}.png").exists() and s in manifest:
            continue
        b = z.read(name)
        img = Image.open(io.BytesIO(b)); fmt = img.format or ""; img = img.convert("RGB"); W, H = img.size
        assert min(W, H) >= a.min_side, (name, W, H)
        scale = CROP / min(W, H)
        rw, rh = (CROP, max(CROP, round(H * scale))) if W <= H else (max(CROP, round(W * scale)), CROP)
        arr = np.array(img)
        if (rw, rh) != (W, H):
            arr = cv2.resize(arr, (rw, rh), interpolation=cv2.INTER_AREA)      # downsample only (scale <= 1), never upsample
        x0 = (rw - CROP) // 2; y0 = (rh - CROP) // 2
        crop = np.ascontiguousarray(arr[y0:y0 + CROP, x0:x0 + CROP])
        view = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "conditions" / "canny2048" / f"{s}.png"), canny3(crop))
        cv2.imwrite(str(out / "conditions" / "canny" / f"{s}.png"), canny3(view))
        cv2.imwrite(str(out / "images" / f"{s}.png"), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "images2048" / f"{s}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))   # last: marks the row done
        manifest[s] = {"sample_id": s, "source_file": name, "sha256": hashlib.sha256(b).hexdigest(), "source_format": fmt, "width": W, "height": H,
                       "resized_width": rw, "resized_height": rh, "crop_x0": x0, "crop_y0": y0, "caption": existing.get(s, {}).get("caption", ""),
                       "dup_group": "", "dev200_idx": "", "subset500": ""}
        n_new += 1
        if n_new % 50 == 0:
            print(f"  [{k+1}/{len(sel)}] {time.time()-t0:.0f}s ({(time.time()-t0)/n_new:.1f}s/img)", flush=True)
            write_manifest(MANIFEST, [manifest[q] for q in sorted(manifest)], COLUMNS)
    write_manifest(MANIFEST, [manifest[q] for q in sorted(manifest)], COLUMNS)
    write_json(REPO_BENCH / SPLIT / "VERSION.json", {
        "split": SPLIT, "benchmark_version": "v1.13", "source": f"hf://datasets/{REPO}/{ZIP} (DIV8K, Gu et al., ICCVW 2019 'DIV8K: DIVerse 8K Resolution Image Dataset')",
        "why_not_lsdir": "LSDIR's HR images are ~1K on the short side (HF mirrors danjacobellis/LSDIR, LSDIR_raw: none >= 2048), checked 2026-09-12",
        "selection": f"members with min(w, h) >= {a.min_side}, seeded shuffle (numpy default_rng({a.seed})), first {a.n}", "n": len(manifest),
        "images2048": "cv2.INTER_AREA downsample so that the short side is 2048 (never upsampled), then the 2048 x 2048 center crop (the real 2048 reference)",
        "images": "cv2.INTER_AREA downsample of the crop to 512 (the real 512 image)",
        "conditions/canny": "c512 = cv2.Canny(gray(512 image), 100, 200)", "conditions/canny2048": "c2048 = cv2.Canny(gray(2048 crop), 100, 200)",
        "reference_of_c2048_column": "the real 2048 image (1.0); the PiD round trip is a row here, not the reference",
        "captions": "one sentence per image by the captioner recorded in CAPTIONS.json (stage --captions)", "env": env_pins()})
    print(f"{SPLIT}: {len(manifest)} rows ({n_new} new) in {time.time()-t0:.0f}s -> {out}; manifest {MANIFEST}")

# ---------------------------------------------------------------- 2. captions: Qwen2.5-VL, one sentence per 512 image
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
