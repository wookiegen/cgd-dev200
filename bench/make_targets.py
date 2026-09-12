"""Build CGD training triplets from a training manifest (BENCHMARK v1.7; paper Method 'Training the Conditioned Decoder').

For every non-blocked row of bench/train/<set>/manifest.csv:
  1. 512x512 source crop x            -> images512/<id>.png            (MultiGen rows are 512 already; ADE20K / COCO use the v1.7 square crop)
  2. clean FLUX latent z0 = E(x)      -> latents/<id>.pt               ((1,16,64,64) fp16 in diffusers' scaled latent space, the space PiD reads)
  3. target y = PiD(z0, sigma=0)       -> targets/<id>.jpg              (2048x2048, condition-blind decode of the CLEAN latent, JPEG q95)
  4. conditions at 512 from the target's area-downsampled view (ControlNet convention, consistent with y by construction):
       conditions/canny/<id>.png       cv2.Canny(gray(area512(y)), 100, 200), 3-channel
       conditions/depth/<id>.png       DPT-Large on area512(y), 8-bit min-max;  conditions/depth_raw/<id>.npy float16 raw
     and, from the GT annotation of the source (coarse structures, unchanged by the decode):
       conditions/seg/<id>.png         ADE20K palette render of the cropped label map   (ade20k_train)
       conditions/bbox/<id>.png        class-color boxes on black after the crop         (coco_train)
The input latent used at TRAIN time (z0 re-noised to tau) is sampled in the training loop, not stored.

Usage (inside the container):  python make_targets.py --set multigen_train30 [--split train|val|all] [--limit N] [--shard k --nshards n]
                               [--batch B] [--no-depth] [--dry-run]   (dry-run: crops + GT conditions only, no models)
Outputs under $CGD_BENCH_ROOT/train/<set>/; a jsonl log per shard in _logs/. Skips rows whose target already exists.
"""
import argparse
import csv
import glob
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 256 * 1024 * 1024   # some MultiGen-20M train PNGs carry huge text chunks; PIL's 1 MB default raises ValueError

from common import (OUT_ROOT, RAW_ROOT, REPO_BENCH, canny_condition, crop_boxes, crop_image, crop_label, parquet_rows, render_boxes,
                    render_palette, sha256_bytes, write_json, env_pins)

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=["multigen_train30", "ade20k_train", "coco_train"])
ap.add_argument("--split", default="all", choices=["train", "val", "all"])
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--batch", type=int, default=1, help="PiD decode batch size (1 is the tested path)")
ap.add_argument("--no-depth", action="store_true")
ap.add_argument("--dry-run", action="store_true", help="no models: write crops and GT-rendered conditions only")
ap.add_argument("--pid", default=os.environ.get("PID_ROOT", str(RAW_ROOT.parent / "PiD")))
ap.add_argument("--pid-ckpt-type", default="2k", help="PiD checkpoint for the TARGETS: the released distilled decoder (baseline rows use the same)")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--jpeg-quality", type=int, default=95)
a = ap.parse_args()

SET = a.set
out = OUT_ROOT / "train" / SET
subdirs = ["images512", "latents", "targets", "conditions/canny", "conditions/canny2048", "conditions/depth", "conditions/depth_raw"]
subdirs += ["conditions/seg"] if SET == "ade20k_train" else ["conditions/bbox"] if SET == "coco_train" else []
for d in subdirs:
    (out / d).mkdir(parents=True, exist_ok=True)
log_dir = RAW_ROOT / "_logs"; log_dir.mkdir(exist_ok=True)
log_path = log_dir / f"targets_{SET}_shard{a.shard}of{a.nshards}.jsonl"

rows = list(csv.DictReader(open(REPO_BENCH / "train" / SET / "manifest.csv")))
sel = [(i, r) for i, r in enumerate(rows) if r["blocked"] != "1" and (a.split == "all" or r["split"] == a.split)]
sel = [(i, r) for (i, r) in sel if i % a.nshards == a.shard]
if a.limit:
    sel = sel[: a.limit]
def _done(r):   # resumable: a dry run is complete for a row once its crop + canny stand-in exist; a real run once its target exists
    s = r["sample_id"]
    if a.dry_run:
        return (out / "images512" / f"{s}.png").exists() and (out / "conditions" / "canny" / f"{s}.png").exists()
    return (out / "targets" / f"{s}.jpg").exists()


todo = {r["sample_id"] for _, r in sel if not _done(r)}
print(f"{SET}: {len(rows)} rows, {len(sel)} selected for shard {a.shard}/{a.nshards}, {len(todo)} to do", flush=True)

# ------------------------------------------------------------------ GT condition helpers per set
palette = None
if SET == "ade20k_train":
    pj = json.load(open(REPO_BENCH / "ade20k_val2k" / "palette.json"))["palette"]
    palette = {int(k): tuple(v) for k, v in pj.items()}
coco_boxes, coco_colors = None, None
if SET == "coco_train":
    cj = json.load(open(REPO_BENCH / "coco_val5k" / "class_colors.json"))["classes"]
    coco_colors = {c["category_id"]: tuple(c["rgb"]) for c in cj}
    inst = json.load(open(RAW_ROOT / "coco2017" / "annotations" / "instances_train2017.json"))
    coco_boxes = defaultdict(list)
    for an in inst["annotations"]:
        if not an.get("iscrowd", 0):
            coco_boxes[an["image_id"]].append((an["bbox"], an["category_id"]))
    del inst


def source_iter():
    """Yield (row, PIL image at 512 crop, gt_condition PIL or None) for the selected rows, in manifest order."""
    want = {r["sample_id"]: r for _, r in sel if r["sample_id"] in todo}
    if SET in ("multigen_train30", "ade20k_train"):
        import pyarrow.parquet as pq
        sub = "multigen_train_subset" if SET == "multigen_train30" else "captioned_ade20k"
        files = sorted(glob.glob(str(RAW_ROOT / sub / "data" / "train-*.parquet")))
        cols = ["image", "text"] if SET == "multigen_train30" else ["image", "prompt", "seg_map"]
        by_file = defaultdict(dict)
        for r in want.values():
            by_file[r["source_file"]][int(r["source_row"])] = r
        for fp in files:
            wanted = by_file.get(Path(fp).name)
            if not wanted:
                continue                                   # nothing from this shard: do not even open it
            pf = pq.ParquetFile(fp); off = 0
            for batch in pf.iter_batches(batch_size=64, columns=cols):
                n_b = batch.num_rows
                hits = [i for i in range(off, off + n_b) if i in wanted]
                for i in hits:                             # convert ONLY the wanted rows (to_pydict on seg_map is what made this slow)
                    r = wanted[i]; k = i - off
                    b = batch.column("image")[k].as_py()["bytes"]
                    assert sha256_bytes(b) == r["sha256"], (SET, r["sample_id"])
                    img = Image.open(io.BytesIO(b)).convert("RGB")
                    gt = None
                    if SET == "ade20k_train":
                        seg_rows = batch.column("seg_map").slice(k, 1).flatten()          # list<uint8> per image row
                        vals = seg_rows.flatten().to_numpy(zero_copy_only=False)
                        seg = vals.reshape(len(seg_rows), -1).astype(np.uint8)
                        img, _ = crop_image(img)
                        gt = render_palette(crop_label(seg), palette)
                    yield r, img, gt
                off += n_b
                if off > max(wanted):
                    break                                  # past the last wanted row of this shard
    else:
        for r in want.values():
            p = RAW_ROOT / "coco2017" / "train2017" / r["file_name"]
            img = Image.open(p).convert("RGB")
            w, h = img.size
            img, _ = crop_image(img)
            bl = coco_boxes.get(int(r["coco_image_id"]), [])
            tb = crop_boxes([b for b, _ in bl], w, h)
            kept = [(b2, c) for (b, c), b2 in zip(bl, tb) if b2 is not None]
            gt = render_boxes([b for b, _ in kept], [c for _, c in kept], coco_colors)
            yield r, img, gt


# ------------------------------------------------------------------ models
if not a.dry_run:
    import cv2
    import torch
    from diffusers import AutoencoderKL
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    dev = "cuda"
    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=torch.bfloat16).to(dev).eval()
    SF, SH = vae.config.scaling_factor, vae.config.shift_factor
    if not a.no_depth:
        dpt_proc = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        dpt = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(dev).eval()
    os.chdir(a.pid); sys.path.insert(0, a.pid)
    from pid._src.inference.checkpoint_registry import get_pid_checkpoint  # noqa: E402
    from pid._src.utils.model_loader import load_model_from_checkpoint    # noqa: E402
    ck = get_pid_checkpoint("flux", a.pid_ckpt_type)
    pid_model, _ = load_model_from_checkpoint(experiment_name=ck.experiment, checkpoint_path=ck.checkpoint_path,
                                              config_file="pid/_src/configs/pid/config.py", enable_fsdp=False, experiment_opts=[], strict=False)
    pid_model.eval()
    cap_key = pid_model.config.input_caption_key
    print("models loaded:", ck.experiment, flush=True)

    @torch.no_grad()
    def encode(imgs):  # list of PIL 512 -> (B,16,64,64) fp16 scaled latents
        x = torch.stack([torch.from_numpy(np.array(im)).permute(2, 0, 1) for im in imgs]).to(dev, torch.bfloat16) / 127.5 - 1.0
        z = vae.encode(x).latent_dist.mode()
        return ((z - SH) * SF).half()

    @torch.no_grad()
    def decode(lats, caps):  # (B,16,64,64) -> list of uint8 HxWx3 at 2048
        batch = {cap_key: list(caps), "LQ_latent": lats.to(dev, torch.bfloat16),
                 "degrade_sigma": torch.zeros(len(caps), device=dev, dtype=torch.float32)}
        hq = (lats.shape[-2] * 8 * 4, lats.shape[-1] * 8 * 4)
        s = pid_model.generate_samples_from_batch(batch, cfg_scale=1.0, num_steps=a.steps, seed=a.seed, shift=None, image_size=hq)
        s = s.float().cpu().clamp(-1, 1)
        if s.ndim == 5:  # (B,3,1,H,W)
            s = s[:, :, 0]
        return [((im.permute(1, 2, 0).numpy() + 1) / 2 * 255).round().clip(0, 255).astype(np.uint8) for im in s]

    @torch.no_grad()
    def depth512(img512):  # PIL 512 -> (raw float32 512x512, uint8 png array)
        inp = dpt_proc(images=img512, return_tensors="pt").to(dev)
        pred = dpt(**inp).predicted_depth
        d = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(512, 512), mode="bicubic", align_corners=False)[0, 0].float().cpu().numpy()
        lo, hi = float(d.min()), float(d.max())
        u8 = np.zeros_like(d, dtype=np.uint8) if hi - lo < 1e-8 else ((d - lo) / (hi - lo) * 255.0).round().astype(np.uint8)
        return d, np.repeat(u8[:, :, None], 3, axis=2)


# ------------------------------------------------------------------ main loop
t0 = time.time(); n = 0; buf = []


def flush(buf):
    global n
    if not buf:
        return
    if a.dry_run:
        for r, img, gt in buf:
            s = r["sample_id"]
            img.save(out / "images512" / f"{s}.png")
            if gt is not None:
                gt.save(out / ("conditions/seg" if SET == "ade20k_train" else "conditions/bbox") / f"{s}.png")
            canny_condition(img).save(out / "conditions" / "canny" / f"{s}.png")   # dry-run stand-in: canny of the SOURCE crop
            n += 1
        return
    import cv2
    lats = encode([img for _, img, _ in buf])
    ys = decode(lats, [r["caption"] for r, _, _ in buf])
    for k, (r, img, gt) in enumerate(buf):
        s = r["sample_id"]
        img.save(out / "images512" / f"{s}.png")
        torch.save({"latent": lats[k:k + 1].cpu(), "sigma": 0.0, "caption": r["caption"], "sample_id": s}, out / "latents" / f"{s}.pt")
        y = ys[k]
        Image.fromarray(y).save(out / "targets" / f"{s}.jpg", quality=a.jpeg_quality, subsampling=0)
        y512 = cv2.resize(y, (512, 512), interpolation=cv2.INTER_AREA)   # the scorer's matched view
        y512_pil = Image.fromarray(y512)
        canny_condition(y512_pil).save(out / "conditions" / "canny" / f"{s}.png")            # 512 condition (default track)
        canny_condition(Image.fromarray(y)).save(out / "conditions" / "canny2048" / f"{s}.png")  # native condition (2026-09-12 track): thin edges of the target at 2048
        if not a.no_depth:
            d, u8 = depth512(y512_pil)
            np.save(out / "conditions" / "depth_raw" / f"{s}.npy", d.astype(np.float16))
            Image.fromarray(u8).save(out / "conditions" / "depth" / f"{s}.png")
        if gt is not None:
            gt.save(out / ("conditions/seg" if SET == "ade20k_train" else "conditions/bbox") / f"{s}.png")
        with open(log_path, "a") as f:
            f.write(json.dumps({"sample_id": s, "t": time.time()}) + "\n")
        n += 1


for r, img, gt in source_iter():
    buf.append((r, img, gt))
    if len(buf) >= a.batch:
        flush(buf); buf = []
        if n % 50 < a.batch:
            print(f"[{n}/{len(todo)}] {time.time()-t0:.0f}s  {(time.time()-t0)/max(n,1):.2f}s/img", flush=True)
flush(buf)
write_json(REPO_BENCH / "train" / SET / ("TARGETS_VERSION.json" if not a.dry_run else "DRYRUN_VERSION.json"), {
    "set": SET, "targets": f"vanilla PiD ({a.pid_ckpt_type if not a.dry_run else 'n/a'}, {a.steps} steps, seed {a.seed}, cfg 1) decode of the CLEAN FLUX latent of the 512 crop, 2048x2048, JPEG q{a.jpeg_quality}",
    "latent": "FLUX.1-dev VAE posterior mean, diffusers scaled space ((z - shift) * scaling), fp16", "conditions": {
        "canny": "cv2.Canny(gray(INTER_AREA 512 view of the target), 100, 200)", "canny2048": "cv2.Canny(gray(target at 2048), 100, 200): the native-condition form (2026-09-12); training samples either form per example",
        "depth": "Intel/dpt-large on the 512 view of the target; png min-max 8-bit, raw float16 npy",
        "seg": "GT ADE20K label map, v1.7 crop, ade20k_val2k/palette.json" if SET == "ade20k_train" else None,
        "bbox": "GT COCO boxes, v1.7 crop, coco_val5k/class_colors.json" if SET == "coco_train" else None},
    "dry_run": a.dry_run, "env": env_pins()})
print(f"done {SET}: {n} rows in {time.time()-t0:.0f}s -> {out}")
