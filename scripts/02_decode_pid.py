#!/usr/bin/env python
"""02: Decode the cached OminiControl latents with vanilla PiD (condition-blind, generative) -> 2048.

Two variants per image, both consuming the SAME cached latent file as the VAE row:
  final : x_0 (sigma 0)          -> the paired decoder-swap row (same latent as the VAE decode)
  et24  : x_t after 24/28 steps  -> PiD's own recommended early-termination operating point (extra row)

PiD: nvidia/PiD `PiD_res2k_sr4x_official_flux_distill_4step` (512-latent -> 2048, 4-step distilled).
The latent is fed as `LQ_latent` exactly as saved (diffusers scaled space; PiD handles VAE normalization),
with `degrade_sigma` = the saved sigma. Outputs:
  outputs/omini_pid/{idx}.png          2048x2048 native
  outputs/omini_pid_512/{idx}.png      matched 512 view, cv2.INTER_AREA (pinned downsampler)
  outputs/omini_pid_et24/{idx}.png     2048 native, early-terminated latent
  outputs/omini_pid_et24_512/{idx}.png matched 512 view

Usage (inside the container, from anywhere):
  CUDA_VISIBLE_DEVICES=1 python scripts/02_decode_pid.py
"""
import argparse, glob, json, os, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CGD_ROOT = os.environ.get("CGD_ROOT", os.path.dirname(REPO))
import numpy as np, cv2, torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=REPO)
ap.add_argument("--pid", default=os.environ.get("PID_ROOT", os.path.join(CGD_ROOT, "PiD")))
ap.add_argument("--ckpt-type", default="2k", choices=["2k", "2kto4k_v1pt5"])
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--scale", type=int, default=4)
ap.add_argument("--load-ema-to-reg", action="store_true")
ap.add_argument("--variants", default="final,et24")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()

os.chdir(a.pid); sys.path.insert(0, a.pid)                      # PiD resolves checkpoints/ae.safetensors relative to its root
from pid._src.inference.checkpoint_registry import get_pid_checkpoint  # noqa: E402
from pid._src.utils.model_loader import load_model_from_checkpoint    # noqa: E402

ck = get_pid_checkpoint("flux", a.ckpt_type)
print("PiD checkpoint:", ck.experiment, ck.checkpoint_path, "scale", ck.pid_scale, flush=True)
model, _ = load_model_from_checkpoint(experiment_name=ck.experiment, checkpoint_path=ck.checkpoint_path,
                                      config_file="pid/_src/configs/pid/config.py", enable_fsdp=False,
                                      experiment_opts=[], strict=False, load_ema_to_reg=a.load_ema_to_reg)
model.eval()
cap_key = model.config.input_caption_key

lat_dir = os.path.join(a.repo, "latents", "omini_canny")
files = sorted(glob.glob(os.path.join(lat_dir, "*.pt")))[: a.limit or None]
variants = a.variants.split(",")
outs = {v: os.path.join(a.repo, "outputs", "omini_pid" + ("" if v == "final" else f"_{v}")) for v in variants}
for v in variants:
    os.makedirs(outs[v], exist_ok=True); os.makedirs(outs[v] + "_512", exist_ok=True)
log_path = os.path.join(a.repo, "results", "log_02_pid.jsonl")

def to_uint8(img):                       # img: (3,H,W) in [-1,1]
    return ((img.permute(1, 2, 0).numpy() + 1) / 2 * 255).round().clip(0, 255).astype(np.uint8)

for n, pt in enumerate(files):
    idx = os.path.splitext(os.path.basename(pt))[0]
    d = torch.load(pt, map_location="cpu")
    for v in variants:
        png = os.path.join(outs[v], f"{idx}.png")
        if os.path.exists(png) and not a.overwrite:
            continue
        lat, sig = (d["latent"], d["sigma"]) if v == "final" else (d["xt24"], d["sigma24"])
        batch = {cap_key: [d["caption"]],
                 "LQ_latent": lat.to(torch.bfloat16).cuda(),
                 "degrade_sigma": torch.tensor([float(sig)], device="cuda", dtype=torch.float32)}
        hq = (lat.shape[-2] * 8 * a.scale, lat.shape[-1] * 8 * a.scale)        # 64*8*4 = 2048
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            s = model.generate_samples_from_batch(batch, cfg_scale=a.cfg, num_steps=a.steps, seed=a.seed,
                                                  shift=None, image_size=hq)
        torch.cuda.synchronize(); dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1e9
        img = s[0].float().cpu().clamp(-1, 1)
        if img.ndim == 4:                    # (3,1,H,W) -> (3,H,W)
            img = img[:, 0]
        arr = to_uint8(img)
        cv2.imwrite(png, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        small = cv2.resize(arr, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(outs[v] + "_512", f"{idx}.png"), cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        with open(log_path, "a") as f:
            f.write(json.dumps({"idx": idx, "variant": v, "sigma": float(sig), "t_dec_s": dt, "peak_mem_gb": mem,
                                "out_hw": list(arr.shape[:2])}) + "\n")
    if n % 10 == 0:
        print(f"[{n+1}/{len(files)}] {idx} {v} {dt:.2f}s mem {mem:.1f}GB out {arr.shape[:2]}", flush=True)
print("done:", outs)
