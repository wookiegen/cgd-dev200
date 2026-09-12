"""Train the conditioned VAE decoder archetype (cond_vae.py) on REAL images: input = FLUX latent of the 512 crop + its condition, target =
the crop itself (L1 + LPIPS). One model per condition. Data = train/<set>/images512 (+ conditions/canny from the dry run of make_targets,
= canny of the source crop; conditions_real/depth from precompute_real_depth.py), rows with split == train and blocked != 1.

Validation every --val-every steps: (a) reconstruction L1 / LPIPS on 64 rows of the val split; (b) the PAPER metric on the canny dev-200:
decode the cached OminiControl latents with their condition and score canny F1 (tolerant) / depth RMSE, the number the 2x2 table reports.
Saves <out>/last.safetensors, <out>/best.safetensors (best paper metric), <out>/log.jsonl.
Usage: CUDA_VISIBLE_DEVICES=g python train_cond_vae.py --condition canny [--steps 6000 --batch 4 --accum 2]
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from common import OUT_ROOT, REPO_BENCH
from cond_vae import CondVAEDecoder, cond_to_tensor

ap = argparse.ArgumentParser()
ap.add_argument("--condition", required=True, choices=["canny", "depth"])
ap.add_argument("--set", default="multigen_train30")
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=4); ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=5e-5, help="decoder lr"); ap.add_argument("--lr-branch", type=float, default=1e-4)
ap.add_argument("--lpips-w", type=float, default=0.5)
ap.add_argument("--val-every", type=int, default=500); ap.add_argument("--n-val", type=int, default=64)
ap.add_argument("--out", default=None); ap.add_argument("--resume", action="store_true"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--freeze-decoder", action="store_true", help="train only the condition branch + injections")
a = ap.parse_args()

dev = "cuda"; torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
OUT = Path(a.out or (OUT_ROOT.parent / "cond_vae" / a.condition)); OUT.mkdir(parents=True, exist_ok=True)
T = OUT_ROOT / "train" / a.set
cond_dir = T / "conditions" / "canny" if a.condition == "canny" else T / "conditions_real" / "depth"

rows = [r for r in csv.DictReader(open(REPO_BENCH / "train" / a.set / "manifest.csv")) if r["blocked"] != "1"]
have = lambda r: (T / "images512" / f"{r['sample_id']}.png").exists() and (cond_dir / f"{r['sample_id']}.png").exists()  # noqa: E731
train_rows = [r for r in rows if r["split"] == "train" and have(r)]
val_rows = [r for r in rows if r["split"] == "val" and have(r)][: a.n_val]
print(f"{a.condition}: {len(train_rows)} train rows, {len(val_rows)} val rows from {T}", flush=True)
assert len(train_rows) > 1000, "training crops / conditions missing (run make_targets.py --dry-run and precompute_real_depth.py first)"


def load_pair(r, flip: bool):
    sid = r["sample_id"]
    img = np.array(Image.open(T / "images512" / f"{sid}.png").convert("RGB"))
    cond = np.array(Image.open(cond_dir / f"{sid}.png").convert("RGB"))
    if flip:
        img = img[:, ::-1]; cond = cond[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(cond)


def batch_tensors(pairs):
    x = torch.from_numpy(np.stack([p[0] for p in pairs])).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)
    c = cond_to_tensor([p[1] for p in pairs], dev)
    return x, c


# ------------------------------------------------------------------ models
from diffusers import AutoencoderKL  # noqa: E402
import pyiqa  # noqa: E402
vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae").to(dev)
vae.requires_grad_(False); vae.eval()
SF, SH = vae.config.scaling_factor, vae.config.shift_factor
model = CondVAEDecoder(vae.decoder, getattr(vae, "post_quant_conv", None)).to(dev)
for p in model.dec.parameters():
    p.requires_grad_(not a.freeze_decoder)
for p in list(model.branch.parameters()) + list(model.inject.parameters()):
    p.requires_grad_(True)
lpips = pyiqa.create_metric("lpips", device=dev, as_loss=True)
groups = [{"params": list(model.branch.parameters()) + list(model.inject.parameters()), "lr": a.lr_branch}]
if not a.freeze_decoder:
    groups.append({"params": list(model.dec.parameters()), "lr": a.lr})
opt = torch.optim.AdamW(groups, weight_decay=1e-2, betas=(0.9, 0.99))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * (0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)) * 0.9 + 0.1))
step0 = 0; best = None
if a.resume and (OUT / "last.pt").exists():
    ck = torch.load(OUT / "last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step0 = ck["step"]; best = ck.get("best")
    print(f"resumed at step {step0}, best {best}", flush=True)


@torch.no_grad()
def encode(x):
    z = vae.encode(x.to(torch.bfloat16) if vae.dtype == torch.bfloat16 else x).latent_dist.mode()
    return z.float()


# ------------------------------------------------------------------ validation: recon on val rows + paper metric on the canny dev-200
dev200 = {}
for r in csv.DictReader(open(REPO_BENCH / "multigen5k" / "manifest.csv")):
    if r["dev200_idx"] and r["dev200_idx"] not in dev200:
        dev200[r["dev200_idx"]] = r["sample_id"]
dev_ids = sorted(dev200.values())
lat_dir = OUT_ROOT / "latents" / "omini" / a.condition
bench_cond = OUT_ROOT / "multigen5k" / "conditions" / a.condition
import scorers as S  # noqa: E402
paper_scorer = S.CannyF1() if a.condition == "canny" else S.DepthScorer()


@torch.no_grad()
def validate():
    model.eval(); out = {}
    l1s, lps = [], []
    for i in range(0, len(val_rows), 8):
        x, c = batch_tensors([load_pair(r, False) for r in val_rows[i:i + 8]])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(encode(x), c).float().clamp(-1, 1)
        l1s.append((y - x).abs().mean().item()); lps.append(lpips((y + 1) / 2, (x + 1) / 2).mean().item())
    out["val_l1"] = float(np.mean(l1s)); out["val_lpips"] = float(np.mean(lps))
    vals = []
    for i in range(0, len(dev_ids), 8):
        ids = dev_ids[i:i + 8]
        lats = torch.cat([torch.load(lat_dir / f"{s}.pt", map_location="cpu")["latent"].float() for s in ids]).to(dev)
        z = lats / SF + SH
        conds = [np.array(Image.open(bench_cond / f"{s}.png").convert("RGB")) for s in ids]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(z, cond_to_tensor(conds, dev)).float().clamp(-1, 1)
        imgs = ((y.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2 * 255).round().clip(0, 255).astype(np.uint8)
        for s, im, cd in zip(ids, imgs, conds):
            if a.condition == "canny":
                vals.append(paper_scorer.score(im, cd)["f1"])
            else:
                ref = np.load(OUT_ROOT / "multigen5k" / "conditions" / "depth_raw" / f"{s}.npy").astype(np.float32)
                vals.append(paper_scorer.score(im, ref)["rmse"])
    out["dev200_" + ("canny_f1" if a.condition == "canny" else "depth_rmse")] = float(np.mean(vals))
    model.train(); return out


def better(m, b):
    if b is None:
        return True
    k = "dev200_canny_f1" if a.condition == "canny" else "dev200_depth_rmse"
    return m[k] > b[k] if a.condition == "canny" else m[k] < b[k]


# ------------------------------------------------------------------ train
model.train(); t0 = time.time(); order = list(range(len(train_rows))); random.shuffle(order); ptr = 0
if step0 == 0:
    m = validate(); m["step"] = 0; print(json.dumps(m), flush=True)
    with open(OUT / "log.jsonl", "a") as f:
        f.write(json.dumps(m) + "\n")
for step in range(step0 + 1, a.steps + 1):
    opt.zero_grad(set_to_none=True); tot = {"l1": 0.0, "lpips": 0.0}
    for _ in range(a.accum):
        if ptr + a.batch > len(order):
            random.shuffle(order); ptr = 0
        pairs = [load_pair(train_rows[j], random.random() < 0.5) for j in order[ptr:ptr + a.batch]]; ptr += a.batch
        x, c = batch_tensors(pairs)
        z = encode(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(z, c)
        y = y.float()
        l1 = (y - x).abs().mean(); lp = lpips((y.clamp(-1, 1) + 1) / 2, (x + 1) / 2).mean()
        loss = (l1 + a.lpips_w * lp) / a.accum
        loss.backward(); tot["l1"] += l1.item() / a.accum; tot["lpips"] += lp.item() / a.accum
    torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
    opt.step(); sched.step()
    if step % 50 == 0:
        print(f"step {step} l1 {tot['l1']:.4f} lpips {tot['lpips']:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
    if step % a.val_every == 0 or step == a.steps:
        m = validate(); m.update({"step": step, "train_l1": tot["l1"], "train_lpips": tot["lpips"], "t_s": round(time.time() - t0)})
        print(json.dumps(m), flush=True)
        with open(OUT / "log.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        save_file({k: v.contiguous() for k, v in model.state_to_save().items()}, str(OUT / "last.safetensors"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "best": best}, OUT / "last.pt")
        if better(m, best):
            best = m; save_file({k: v.contiguous() for k, v in model.state_to_save().items()}, str(OUT / "best.safetensors"))
            json.dump(best, open(OUT / "best.json", "w"), indent=1)
print("done; best", best)
