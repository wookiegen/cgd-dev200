"""Generate the conditioned FLUX.1-dev latents of an eval set under one controller (BENCHMARK Sections 2, 5): the paired inputs every
decoder row consumes, plus the controller's own VAE-decode row.

Controllers (all at 512x512, 28 steps, guidance 3.5, seed 0, prompt = manifest caption, condition = our pinned annotator's map):
  omini        OminiControl released LoRAs (Yuanshi/OminiControl experimental/{canny,depth}.safetensors), token-concat
  easycontrol  EasyControl released LoRAs (Xiaojiu-Z/EasyControl models/{canny,depth,seg}.safetensors), token-concat with KV cache;
               cond_size 512 (its training condition size), LoRA weight 1; the repo's custom pipeline ($EASYCONTROL_ROOT)
  fluxcn       FLUX ControlNet Union-Pro-2.0 (Shakker-Labs), diffusers FluxControlNetPipeline; published settings canny scale 0.7,
               depth scale 0.8, control_guidance_end 0.8 (no control_mode in Pro-2.0)
Saved per sample, keyed by sample_id:
  latents/<controller>/<condition>/<id>.pt          x_0 (1,16,64,64) fp16 (diffusers scaled space), sigma 0; xt16/xt24 + sigma16/sigma24
  outputs/<controller>/<condition>/vae@28/<id>.png  the VAE decode of x_0 (the controller as published), 512
Usage: CUDA_VISIBLE_DEVICES=g python make_latents.py --controller <omini|easycontrol|fluxcn> --condition <canny|depth|seg> --split <set> --shard g --nshards n
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
ap.add_argument("--controller", default="omini", choices=["omini", "easycontrol", "fluxcn"])
ap.add_argument("--condition", required=True, choices=["canny", "depth", "seg"])
ap.add_argument("--split", default="multigen5k")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--steps", type=int, default=28); ap.add_argument("--guidance", type=float, default=3.5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--capture-steps", default="16,24")
ap.add_argument("--omini", default=os.environ.get("OMINI_ROOT", str(OUT_ROOT.parent.parent / "OminiControl")))
ap.add_argument("--easycontrol", default=os.environ.get("EASYCONTROL_ROOT", str(OUT_ROOT.parent.parent / "EasyControl")))
ap.add_argument("--out-tag", default=None, help="override the output directory name (default = controller)")
ap.add_argument("--size", type=int, default=512, help="generation resolution (OPEN_QUESTIONS 11 side comparison: 1024 for EasyControl / ControlNet); the 512 condition is resized to it (nearest for canny / seg, bicubic for depth)")
ap.add_argument("--subset500", action="store_true", help="only the manifest rows with subset500 == 1")
a = ap.parse_args()

rows = list(csv.DictReader(open(REPO_BENCH / a.split / "manifest.csv")))
if a.subset500:
    rows = [r for r in rows if r.get("subset500") == "1"]
rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard][: a.limit or None]
cond_dir = OUT_ROOT / a.split / "conditions" / a.condition
tag = a.out_tag or a.controller
d_lat = OUT_ROOT / "latents" / tag / a.condition
d_vae = OUT_ROOT / "outputs" / tag / a.condition / "vae@28"
for d in [d_lat, d_vae]:
    d.mkdir(parents=True, exist_ok=True)
log = OUT_ROOT / "latents" / tag / f"log_{a.condition}_{a.split}_shard{a.shard}.jsonl"
dev = "cuda"; H = W = a.size
COND_RESAMPLE = Image.BICUBIC if a.condition == "depth" else Image.NEAREST
VSF = 8   # FLUX VAE downsampling factor for _unpack_latents (EasyControl's vendored pipeline reports a different vae_scale_factor convention)
steps_cap = [int(x) for x in a.capture_steps.split(",")]
from diffusers.pipelines import FluxPipeline  # noqa: E402

# ------------------------------------------------------------------ controller-specific setup: returns (pipe, run(prompt, cond_img, generator, cb) -> packed latents)
if a.controller == "omini":
    sys.path.insert(0, a.omini)
    from omini.pipeline.flux_omini import Condition, generate  # noqa: E402
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(dev)
    LORA = {"canny": "experimental/canny.safetensors", "depth": "experimental/depth.safetensors"}[a.condition]
    pipe.load_lora_weights("Yuanshi/OminiControl", weight_name=LORA, adapter_name=a.condition); pipe.set_adapters([a.condition])
    settings = {"lora": LORA}

    def run(prompt, cond, g, cb):
        return generate(pipe, prompt=prompt, conditions=[Condition(cond, a.condition)], height=H, width=W, num_inference_steps=a.steps,
                        guidance_scale=a.guidance, generator=g, output_type="latent", callback_on_step_end=cb).images

elif a.controller == "easycontrol":
    sys.path.insert(0, a.easycontrol)
    from huggingface_hub import hf_hub_download  # noqa: E402
    from src.pipeline import FluxPipeline as EasyFluxPipeline  # noqa: E402
    from src.transformer_flux import FluxTransformer2DModel as EasyTransformer  # noqa: E402
    from src.lora_helper import set_single_lora  # noqa: E402
    pipe = EasyFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.transformer = EasyTransformer.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe.to(dev)
    lora_path = hf_hub_download("Xiaojiu-Z/EasyControl", f"models/{a.condition}.safetensors")
    set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_size=512)
    settings = {"lora": f"models/{a.condition}.safetensors", "cond_size": 512, "lora_weight": 1, "max_sequence_length": 512}

    def clear_cache():
        for _, p in pipe.transformer.attn_processors.items():
            if hasattr(p, "bank_kv"):
                p.bank_kv.clear()

    def run(prompt, cond, g, cb):
        out = pipe(prompt, height=H, width=W, guidance_scale=a.guidance, num_inference_steps=a.steps, max_sequence_length=512, generator=g,
                   spatial_images=[cond], subject_images=[], cond_size=512, output_type="latent", callback_on_step_end=cb).images
        clear_cache()
        return out

elif a.controller == "fluxcn":
    from diffusers import FluxControlNetModel, FluxControlNetPipeline  # noqa: E402
    cn = FluxControlNetModel.from_pretrained("Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0", torch_dtype=torch.bfloat16)
    pipe = FluxControlNetPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", controlnet=cn, torch_dtype=torch.bfloat16).to(dev)
    SCALE = {"canny": 0.7, "depth": 0.8}[a.condition]
    settings = {"controlnet_conditioning_scale": SCALE, "control_guidance_end": 0.8, "note": "Union-Pro-2.0 needs no control_mode"}

    def run(prompt, cond, g, cb):
        return pipe(prompt, control_image=cond, height=H, width=W, controlnet_conditioning_scale=SCALE, control_guidance_end=0.8,
                    num_inference_steps=a.steps, guidance_scale=a.guidance, generator=g, output_type="latent", callback_on_step_end=cb).images

pipe.set_progress_bar_config(disable=True)
print(f"{tag}/{a.condition} on {a.split}: shard {a.shard}/{a.nshards}, {len(rows)} rows, {settings}", flush=True)

# ------------------------------------------------------------------ main loop
t0 = time.time(); n = 0
for r in rows:
    sid = r["sample_id"]
    pt = d_lat / f"{sid}.pt"
    if pt.exists() and (d_vae / f"{sid}.png").exists():
        continue
    cond = Image.open(cond_dir / f"{sid}.png").convert("RGB")
    if cond.size != (W, H):
        cond = cond.resize((W, H), COND_RESAMPLE)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    captured = {}

    def cb(pipeline, i, t, kw):
        if (i + 1) in steps_cap:
            captured[i + 1] = (kw["latents"].detach().clone(), float(pipeline.scheduler.sigmas[i + 1].item()))
        return kw

    t1 = time.time()
    with torch.no_grad():
        packed = run(r["caption"], cond, g, cb)
        lat = FluxPipeline._unpack_latents(packed, H, W, VSF)
        z = lat.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        img = pipe.image_processor.postprocess(pipe.vae.decode(z, return_dict=False)[0], output_type="pil")[0]
    torch.cuda.synchronize(); t_gen = time.time() - t1
    rec = {"latent": lat.half().cpu(), "sigma": float(pipe.scheduler.sigmas[-1].item()), "capture_steps": steps_cap, "caption": r["caption"],
           "seed": a.seed, "steps": a.steps, "guidance": a.guidance, "controller": a.controller, "condition": a.condition, "sample_id": sid,
           "size": a.size, "t_gen_s": t_gen, "settings": settings}
    for K, (x, sg) in captured.items():
        rec[f"xt{K}"] = FluxPipeline._unpack_latents(x, H, W, VSF).half().cpu(); rec[f"sigma{K}"] = sg
    torch.save(rec, pt)
    img.save(d_vae / f"{sid}.png")
    with open(log, "a") as f:
        f.write(json.dumps({"sample_id": sid, "t_gen_s": t_gen, "sigmas": {K: sg for K, (_, sg) in captured.items()}}) + "\n")
    n += 1
    if n % 50 == 0:
        print(f"[{n}/{len(rows)}] {time.time()-t0:.0f}s ({(time.time()-t0)/n:.2f}s/img)", flush=True)
print(f"done {tag}/{a.condition} shard {a.shard}: {n} new in {time.time()-t0:.0f}s")
