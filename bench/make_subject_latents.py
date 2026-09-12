"""Generate the subject-conditioned FLUX.1-dev latents of dreambench750 under one controller's released SUBJECT LoRA (tab:subject rows).

  omini        Yuanshi/OminiControl omini/subject_512.safetensors, Condition(ref, "subject", position_delta=(0, 32)), FLUX.1-dev settings of
               examples/subject_dev.ipynb: guidance 3.5 and image_guidance_scale 1.5 (real-image CFG, required on dev); 28 steps, seed 0
  easycontrol  Xiaojiu-Z/EasyControl models/subject.safetensors, subject_images=[ref], cond_size 512, LoRA weight 1; guidance 3.5, 28 steps
Reference = the set's reference image of the subject (first DreamBooth image, square), resized to 512. Prompt = the manifest prompt
(DreamBench class-name prompt, e.g. "a backpack in the jungle"; no unique token).
Saved per sample: latents/<controller>/subject/<id>.pt (x_0 + xt16/xt24 + sigmas), outputs/<controller>/subject/vae@28/<id>.png (512).
Then: decode_pid.py --controller <c> --condition subject; harness.py --split dreambench750 --condition subject.
Usage: CUDA_VISIBLE_DEVICES=g python make_subject_latents.py --controller omini --shard g --nshards n
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
ap.add_argument("--controller", default="omini", choices=["omini", "easycontrol"])
ap.add_argument("--split", default="dreambench750")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--steps", type=int, default=28); ap.add_argument("--guidance", type=float, default=3.5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--image-guidance", type=float, default=1.5, help="OminiControl real-image CFG on FLUX.1-dev (must be > 1)")
ap.add_argument("--capture-steps", default="16,24")
ap.add_argument("--omini", default=os.environ.get("OMINI_ROOT", str(OUT_ROOT.parent.parent / "OminiControl")))
ap.add_argument("--easycontrol", default=os.environ.get("EASYCONTROL_ROOT", str(OUT_ROOT.parent.parent / "EasyControl")))
ap.add_argument("--prompt-style", default="class", choices=["class", "item"],
                help="class = the DreamBench prompt as in the manifest ('a backpack in the jungle'); item = OminiControl's recommended phrasing, "
                     "'a {0} {1}' -> 'this item' ('this item in the jungle'; 'a red {0} {1}' -> 'this red item')")
ap.add_argument("--out-tag", default=None, help="output directory name under latents/ and outputs/ (default = controller)")
a = ap.parse_args()

import re  # noqa: E402


def prompt_of(r):
    if a.prompt_style == "class":
        return r["prompt"]
    p = re.sub(r"\b[Aa] (\w+ )?\{0\} \{1\}", lambda m: "this item" if not m.group(1) else f"this {m.group(1)}item", r["prompt_template"])
    p = p.replace("{0} {1}", "this item").replace("{0}", "").replace("{1}", "this item")
    return re.sub(r"\s+", " ", p).strip()


rows = list(csv.DictReader(open(REPO_BENCH / a.split / "manifest.csv")))
rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard][: a.limit or None]
ref_dir = OUT_ROOT / a.split / "refs"
tag = a.out_tag or a.controller
d_lat = OUT_ROOT / "latents" / tag / "subject"
d_vae = OUT_ROOT / "outputs" / tag / "subject" / "vae@28"
for d in [d_lat, d_vae]:
    d.mkdir(parents=True, exist_ok=True)
log = OUT_ROOT / "latents" / tag / f"log_subject_{a.split}_shard{a.shard}.jsonl"
dev = "cuda"; H = W = 512; VSF = 8
steps_cap = [int(x) for x in a.capture_steps.split(",")]
from diffusers.pipelines import FluxPipeline  # noqa: E402


def load_ref(r):
    p = ref_dir / f"{r['subject']}{os.path.splitext(r['ref_image'])[1].lower()}"
    im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))   # square already for DreamBench; center crop for safety
    return im.resize((512, 512), Image.LANCZOS)


if a.controller == "omini":
    sys.path.insert(0, a.omini)
    from omini.pipeline.flux_omini import Condition, generate  # noqa: E402
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(dev)
    pipe.load_lora_weights("Yuanshi/OminiControl", weight_name="omini/subject_512.safetensors", adapter_name="subject"); pipe.set_adapters(["subject"])
    settings = {"lora": "omini/subject_512.safetensors", "position_delta": [0, 32], "image_guidance_scale": a.image_guidance, "prompt_style": a.prompt_style}

    def run(prompt, ref, g, cb):
        return generate(pipe, prompt=prompt, conditions=[Condition(ref, "subject", position_delta=(0, 32))], height=H, width=W,
                        num_inference_steps=a.steps, guidance_scale=a.guidance, image_guidance_scale=a.image_guidance, generator=g,
                        output_type="latent", callback_on_step_end=cb).images

elif a.controller == "easycontrol":
    sys.path.insert(0, a.easycontrol)
    from huggingface_hub import hf_hub_download  # noqa: E402
    from src.pipeline import FluxPipeline as EasyFluxPipeline  # noqa: E402
    from src.transformer_flux import FluxTransformer2DModel as EasyTransformer  # noqa: E402
    from src.lora_helper import set_single_lora  # noqa: E402
    pipe = EasyFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.transformer = EasyTransformer.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe.to(dev)
    lora_path = hf_hub_download("Xiaojiu-Z/EasyControl", "models/subject.safetensors")
    set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_size=512)
    settings = {"lora": "models/subject.safetensors", "cond_size": 512, "lora_weight": 1, "max_sequence_length": 512, "prompt_style": a.prompt_style}

    def clear_cache():
        for _, p in pipe.transformer.attn_processors.items():
            if hasattr(p, "bank_kv"):
                p.bank_kv.clear()

    def run(prompt, ref, g, cb):
        out = pipe(prompt, height=H, width=W, guidance_scale=a.guidance, num_inference_steps=a.steps, max_sequence_length=512, generator=g,
                   spatial_images=[], subject_images=[ref], cond_size=512, output_type="latent", callback_on_step_end=cb).images
        clear_cache()
        return out

pipe.set_progress_bar_config(disable=True)
print(f"{tag}/subject on {a.split}: shard {a.shard}/{a.nshards}, {len(rows)} rows, {settings}; e.g. prompt '{prompt_of(rows[0])}'", flush=True)
t0 = time.time(); n = 0
for r in rows:
    sid = r["sample_id"]; pt = d_lat / f"{sid}.pt"
    if pt.exists() and (d_vae / f"{sid}.png").exists():
        continue
    ref = load_ref(r); prompt = prompt_of(r)
    g = torch.Generator(device=dev).manual_seed(a.seed); captured = {}

    def cb(pipeline, i, t, kw):
        if (i + 1) in steps_cap:
            captured[i + 1] = (kw["latents"].detach().clone(), float(pipeline.scheduler.sigmas[i + 1].item()))
        return kw

    t1 = time.time()
    with torch.no_grad():
        packed = run(prompt, ref, g, cb)
        lat = FluxPipeline._unpack_latents(packed, H, W, VSF)
        z = lat.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        img = pipe.image_processor.postprocess(pipe.vae.decode(z, return_dict=False)[0], output_type="pil")[0]
    torch.cuda.synchronize(); t_gen = time.time() - t1
    rec = {"latent": lat.half().cpu(), "sigma": float(pipe.scheduler.sigmas[-1].item()), "capture_steps": steps_cap, "caption": prompt, "protocol_prompt": r["prompt"],
           "seed": a.seed, "steps": a.steps, "guidance": a.guidance, "controller": a.controller, "condition": "subject", "sample_id": sid,
           "subject": r["subject"], "t_gen_s": t_gen, "settings": settings}
    for K, (x, sg) in captured.items():
        rec[f"xt{K}"] = FluxPipeline._unpack_latents(x, H, W, VSF).half().cpu(); rec[f"sigma{K}"] = sg
    torch.save(rec, pt); img.save(d_vae / f"{sid}.png")
    with open(log, "a") as f:
        f.write(json.dumps({"sample_id": sid, "t_gen_s": t_gen, "sigmas": {K: sg for K, (_, sg) in captured.items()}}) + "\n")
    n += 1
    if n % 50 == 0:
        print(f"[{n}/{len(rows)}] {time.time()-t0:.0f}s ({(time.time()-t0)/n:.2f}s/img)", flush=True)
print(f"done {a.controller}/subject shard {a.shard}: {n} new in {time.time()-t0:.0f}s")
