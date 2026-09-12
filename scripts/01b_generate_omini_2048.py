#!/usr/bin/env python
"""01b: The NATIVE ROUTE for the VAE row on dev-200: OminiControl (canny LoRA) generating at 2048x2048 DIRECTLY, then the VAE decode.

This is what "just generate at the output resolution" gives, as opposed to the interpolation route (bicubic x4 of the 512 output, the
dagger value) and the pixel-decoder route (PiD / CGD from the 512 latent). FLUX.1-dev at 4 MP is outside its training range and the canny
LoRA was trained at 512, so quality may drop; that is the honest result of this route. Same prompts, seed 0, 28 steps, guidance 3.5; the
512 condition is nearest-resized to 2048.
  outputs/omini_vae_gen2048/{idx}.png       2048 (native);   outputs/omini_vae_gen2048_512/{idx}.png   INTER_AREA view
Usage: CUDA_VISIBLE_DEVICES=g python scripts/01b_generate_omini_2048.py [--shard g --nshards n] [--limit N]
"""
import argparse, json, os, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CGD_ROOT = os.path.dirname(REPO)
import cv2, numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=REPO)
ap.add_argument("--omini", default=os.environ.get("OMINI_ROOT", os.path.join(CGD_ROOT, "OminiControl")))
ap.add_argument("--size", type=int, default=2048)
ap.add_argument("--steps", type=int, default=28); ap.add_argument("--guidance", type=float, default=3.5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.omini)
from diffusers.pipelines import FluxPipeline  # noqa: E402
from omini.pipeline.flux_omini import Condition, generate  # noqa: E402

dev = os.path.join(a.repo, "dev200"); caps = json.load(open(os.path.join(dev, "captions.json")))
ids = sorted(caps); ids = [i for k, i in enumerate(ids) if k % a.nshards == a.shard][: a.limit or None]
tag = "omini_vae_gen2048" if a.size == 2048 else f"omini_vae_gen{a.size}"
out = os.path.join(a.repo, "outputs", tag); out512 = out + "_512"
os.makedirs(out, exist_ok=True); os.makedirs(out512, exist_ok=True)
log = os.path.join(a.repo, "results", "log_01b_gen2048.jsonl"); os.makedirs(os.path.dirname(log), exist_ok=True)
device = "cuda"; H = W = a.size; VSF = 8
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
pipe.load_lora_weights("Yuanshi/OminiControl", weight_name="experimental/canny.safetensors", adapter_name="canny"); pipe.set_adapters(["canny"])
pipe.set_progress_bar_config(disable=True)
print(f"{tag}: {len(ids)} images at {a.size}", flush=True)
t0 = time.time(); n = 0
for idx in ids:
    p = os.path.join(out, f"{idx}.png")
    if os.path.exists(p) and not a.overwrite:
        continue
    cond = Image.open(os.path.join(dev, "canny", f"{idx}.png")).convert("RGB").resize((W, H), Image.NEAREST)
    g = torch.Generator(device=device).manual_seed(a.seed)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t1 = time.time()
    with torch.no_grad():
        packed = generate(pipe, prompt=caps[idx], conditions=[Condition(cond, "canny")], height=H, width=W, num_inference_steps=a.steps,
                          guidance_scale=a.guidance, generator=g, output_type="latent").images
        lat = FluxPipeline._unpack_latents(packed, H, W, VSF)
        z = lat.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        img = pipe.image_processor.postprocess(pipe.vae.decode(z, return_dict=False)[0], output_type="pil")[0]
    torch.cuda.synchronize(); dt = time.time() - t1; mem = torch.cuda.max_memory_allocated() / 1e9
    img.save(p)
    y = np.array(img); cv2.imwrite(os.path.join(out512, f"{idx}.png"), cv2.cvtColor(cv2.resize(y, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
    with open(log, "a") as f:
        f.write(json.dumps({"idx": idx, "size": a.size, "t_gen_vae_s": dt, "peak_mem_gb": mem}) + "\n")
    n += 1
    if n % 10 == 0:
        print(f"[{n}/{len(ids)}] {time.time()-t0:.0f}s ({(time.time()-t0)/n:.1f}s/img)", flush=True)
print(f"done shard {a.shard}: {n} in {time.time()-t0:.0f}s")
