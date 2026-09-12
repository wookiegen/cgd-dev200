"""Decode a controller's cached latents with the trained conditioned VAE decoder (the aware / deterministic row of tab:decoder-conditioning).
Reads latents/<controller>/<condition>/<sid>.pt (diffusers-scaled x_0) and the eval condition <split>/conditions/<condition>/<sid>.png;
writes outputs/<controller>/<condition>/condvae/<sid>.png (512). Then: harness.py --method condvae --controller <c> --split <split> --condition <cond> --res 512
Usage: CUDA_VISIBLE_DEVICES=g python decode_cond_vae.py --condition canny --controller omini [--ckpt <path>] [--split multigen5k]
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
from PIL import Image

from common import OUT_ROOT
from cond_vae import cond_to_tensor, load_cond_vae

ap = argparse.ArgumentParser()
ap.add_argument("--condition", required=True); ap.add_argument("--controller", default="omini"); ap.add_argument("--split", default="multigen5k")
ap.add_argument("--ckpt", default=None); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out-name", default="condvae")
a = ap.parse_args()
ckpt = a.ckpt or str(OUT_ROOT.parent / "cond_vae" / a.condition / "best.safetensors")
d_lat = OUT_ROOT / "latents" / a.controller / a.condition
cond_dir = OUT_ROOT / a.split / "conditions" / a.condition
out = OUT_ROOT / "outputs" / a.controller / a.condition / a.out_name; out.mkdir(parents=True, exist_ok=True)
files = sorted(glob.glob(str(d_lat / "*.pt")))[: a.limit or None]
files = [f for f in files if not (out / (os.path.splitext(os.path.basename(f))[0] + ".png")).exists()]
dev = "cuda"
from diffusers import AutoencoderKL  # noqa: E402
vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae").to(dev).eval()
SF, SH = vae.config.scaling_factor, vae.config.shift_factor
model = load_cond_vae(vae, ckpt, dev).eval()
print(f"condvae {a.controller}/{a.condition}: {len(files)} latents, ckpt {ckpt}", flush=True)
t0 = time.time()
for i in range(0, len(files), a.batch):
    fs = files[i:i + a.batch]; ids = [os.path.splitext(os.path.basename(f))[0] for f in fs]
    lats = torch.cat([torch.load(f, map_location="cpu")["latent"].float() for f in fs]).to(dev)
    conds = [np.array(Image.open(cond_dir / f"{s}.png").convert("RGB")) for s in ids]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(lats / SF + SH, cond_to_tensor(conds, dev)).float().clamp(-1, 1)
    imgs = ((y.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2 * 255).round().clip(0, 255).astype(np.uint8)
    for s, im in zip(ids, imgs):
        Image.fromarray(im).save(out / f"{s}.png")
    if (i // a.batch) % 50 == 0:
        print(f"  [{i+len(fs)}/{len(files)}] {time.time()-t0:.0f}s", flush=True)
print(f"done {len(files)} in {time.time()-t0:.0f}s -> {out}")
