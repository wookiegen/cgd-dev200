"""Generate the conditioned FLUX.1-dev latents of an eval set under one controller (BENCHMARK Sections 2, 5): the paired inputs every
decoder row consumes, plus the controller's own VAE-decode row.

Controller: OminiControl (released canny / depth LoRAs). Protocol: 512x512, 28 steps, guidance 3.5, seed 0, prompt = manifest caption,
condition = $CGD_BENCH_ROOT/<split>/conditions/<condition>/<id>.png (our pinned annotator). Saved per sample, keyed by sample_id:
  latents/<controller>/<condition>/<id>.pt        latent x_0 (1,16,64,64) fp16 (diffusers scaled space), sigma 0; xt16/xt24 + sigma16/sigma24
                                                  (the latent after K of 28 steps, for the truncation rows); caption, seed, steps, guidance, t_gen_s
  outputs/<controller>/<condition>/vae@28/<id>.png  the VAE decode of x_0 (the controller as published), 512
Usage (inside the container): CUDA_VISIBLE_DEVICES=g python make_latents.py --controller omini --condition canny --split multigen5k --shard g --nshards n
"""
import argparse
import csv
import json
import os
import sys
import time

import torch
from PIL import Image

from common import OUT_ROOT, REPO_BENCH

ap = argparse.ArgumentParser()
ap.add_argument("--controller", default="omini", choices=["omini"])
ap.add_argument("--condition", required=True, choices=["canny", "depth"])
ap.add_argument("--split", default="multigen5k")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--steps", type=int, default=28); ap.add_argument("--guidance", type=float, default=3.5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--capture-steps", default="16,24")
ap.add_argument("--omini", default=os.environ.get("OMINI_ROOT", str(OUT_ROOT.parent.parent / "OminiControl")))
a = ap.parse_args()

rows = list(csv.DictReader(open(REPO_BENCH / a.split / "manifest.csv")))
rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard][: a.limit or None]
cond_dir = OUT_ROOT / a.split / "conditions" / a.condition
d_lat = OUT_ROOT / "latents" / a.controller / a.condition
d_vae = OUT_ROOT / "outputs" / a.controller / a.condition / "vae@28"
for d in [d_lat, d_vae]:
    d.mkdir(parents=True, exist_ok=True)
log = OUT_ROOT / "latents" / a.controller / f"log_{a.condition}_{a.split}_shard{a.shard}.jsonl"

sys.path.insert(0, a.omini)
from diffusers.pipelines import FluxPipeline  # noqa: E402
from omini.pipeline.flux_omini import Condition, generate  # noqa: E402

dev = "cuda"
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(dev)
LORA = {"canny": "experimental/canny.safetensors", "depth": "experimental/depth.safetensors"}[a.condition]
pipe.load_lora_weights("Yuanshi/OminiControl", weight_name=LORA, adapter_name=a.condition)
pipe.set_adapters([a.condition])
pipe.set_progress_bar_config(disable=True)
H = W = 512
steps_cap = [int(x) for x in a.capture_steps.split(",")]
print(f"{a.controller}/{a.condition} on {a.split}: shard {a.shard}/{a.nshards}, {len(rows)} rows, LoRA {LORA}", flush=True)

t0 = time.time(); n = 0
for r in rows:
    sid = r["sample_id"]
    pt = d_lat / f"{sid}.pt"
    if pt.exists() and (d_vae / f"{sid}.png").exists():
        continue
    cond = Image.open(cond_dir / f"{sid}.png").convert("RGB")
    g = torch.Generator(device=dev).manual_seed(a.seed)
    captured = {}

    def cb(pipeline, i, t, kw):
        if (i + 1) in steps_cap:
            captured[i + 1] = (kw["latents"].detach().clone(), float(pipeline.scheduler.sigmas[i + 1].item()))
        return kw

    t1 = time.time()
    with torch.no_grad():
        out = generate(pipe, prompt=r["caption"], conditions=[Condition(cond, a.condition)], height=H, width=W,
                       num_inference_steps=a.steps, guidance_scale=a.guidance, generator=g, output_type="latent", callback_on_step_end=cb)
        lat = FluxPipeline._unpack_latents(out.images, H, W, pipe.vae_scale_factor)
        z = lat.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        img = pipe.image_processor.postprocess(pipe.vae.decode(z, return_dict=False)[0], output_type="pil")[0]
    torch.cuda.synchronize(); t_gen = time.time() - t1
    rec = {"latent": lat.half().cpu(), "sigma": float(pipe.scheduler.sigmas[-1].item()), "capture_steps": steps_cap, "caption": r["caption"],
           "seed": a.seed, "steps": a.steps, "guidance": a.guidance, "controller": a.controller, "condition": a.condition, "sample_id": sid, "t_gen_s": t_gen}
    for K, (x, sg) in captured.items():
        rec[f"xt{K}"] = FluxPipeline._unpack_latents(x, H, W, pipe.vae_scale_factor).half().cpu(); rec[f"sigma{K}"] = sg
    torch.save(rec, pt)
    img.save(d_vae / f"{sid}.png")
    with open(log, "a") as f:
        f.write(json.dumps({"sample_id": sid, "t_gen_s": t_gen, "sigmas": {K: sg for K, (_, sg) in captured.items()}}) + "\n")
    n += 1
    if n % 50 == 0:
        print(f"[{n}/{len(rows)}] {time.time()-t0:.0f}s ({(time.time()-t0)/n:.2f}s/img)", flush=True)
print(f"done {a.controller}/{a.condition} shard {a.shard}: {n} new in {time.time()-t0:.0f}s")
