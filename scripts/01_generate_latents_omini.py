#!/usr/bin/env python
"""01: Generate OminiControl(canny)-conditioned FLUX.1-dev latents for dev-200, plus the VAE-decode row.

Protocol (BENCHMARK v1.3 / DEV_SETTING):
  generator  FLUX.1-dev, OminiControl canny LoRA (Yuanshi/OminiControl experimental/canny.safetensors)
  resolution 512x512 (latent 16 x 64 x 64), 28 steps, guidance 3.5, seed 0 (torch.Generator per image)
  prompt     the MultiGen caption
  condition  dev200/canny/{idx}.png  (cv2.Canny(gray, 100, 200), RGB-replicated)

For every image we save ONE cached latent file that every decoder consumes (paired comparison):
  latents/omini_canny/{idx}.pt :
    latent   (1,16,64,64) fp16  final clean latent x_0, in diffusers' scaled latent space
                                (this is exactly what PiD's `extract_latent` produces; PiD handles VAE scaling)
    sigma    float               scheduler sigma for x_0 (0.0)
    xt24     (1,16,64,64) fp16  the partially-denoised latent after --capture-step steps (PiD's early-termination point)
    sigma24  float               its sigma (residual noise level), consumed by PiD's sigma-aware adapter
    caption, seed, steps, guidance, t_gen_s
and the standard condition-blind deterministic baseline:
  outputs/omini_vae/{idx}.png     VAE decode of the final latent, 512x512

Usage (inside the container, GPU 0):
  CUDA_VISIBLE_DEVICES=0 python scripts/01_generate_latents_omini.py
"""
import argparse, json, os, sys, time
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/data/wookiekim/cgd/cgd-dev200")
ap.add_argument("--omini", default="/data/wookiekim/cgd/OminiControl", help="clean OminiControl clone (provides `omini`)")
ap.add_argument("--steps", type=int, default=28)
ap.add_argument("--guidance", type=float, default=3.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--capture-step", type=int, default=24, help="save x_t after this many steps (PiD early-termination point)")
ap.add_argument("--limit", type=int, default=0, help="debug: only the first N images")
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()

sys.path.insert(0, a.omini)
from diffusers.pipelines import FluxPipeline  # noqa: E402
from omini.pipeline.flux_omini import Condition, generate  # noqa: E402

dev = os.path.join(a.repo, "dev200")
out_lat = os.path.join(a.repo, "latents", "omini_canny"); os.makedirs(out_lat, exist_ok=True)
out_vae = os.path.join(a.repo, "outputs", "omini_vae"); os.makedirs(out_vae, exist_ok=True)
log_path = os.path.join(a.repo, "results", "log_01_generate.jsonl"); os.makedirs(os.path.dirname(log_path), exist_ok=True)

caps = json.load(open(os.path.join(dev, "captions.json")))
ids = sorted(caps)[: a.limit or None]

device = "cuda"
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
pipe.load_lora_weights("Yuanshi/OminiControl", weight_name="experimental/canny.safetensors", adapter_name="canny")
pipe.set_adapters(["canny"])
pipe.set_progress_bar_config(disable=True)
print(f"loaded FLUX.1-dev + OminiControl canny LoRA; {len(ids)} images; steps={a.steps} guidance={a.guidance} seed={a.seed}", flush=True)

H = W = 512
for n, idx in enumerate(ids):
    pt = os.path.join(out_lat, f"{idx}.pt")
    if os.path.exists(pt) and not a.overwrite:
        continue
    cap = caps[idx]
    cond = Image.open(os.path.join(dev, "canny", f"{idx}.png")).convert("RGB")
    g = torch.Generator(device=device).manual_seed(a.seed)
    captured = {}

    def cb(pipeline, i, t, kw):
        if i + 1 == a.capture_step:
            captured["xt"] = kw["latents"].detach().clone()
            captured["sigma"] = float(pipeline.scheduler.sigmas[i + 1].item())
        return kw

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = generate(pipe, prompt=cap, conditions=[Condition(cond, "canny")],
                       height=H, width=W, num_inference_steps=a.steps, guidance_scale=a.guidance,
                       generator=g, output_type="latent", callback_on_step_end=cb)
    torch.cuda.synchronize(); t_gen = time.time() - t0
    packed = out.images                                                   # (1, 1024, 64) packed
    lat = FluxPipeline._unpack_latents(packed, H, W, pipe.vae_scale_factor)   # (1, 16, 64, 64)
    xt = FluxPipeline._unpack_latents(captured["xt"], H, W, pipe.vae_scale_factor)
    final_sigma = float(pipe.scheduler.sigmas[-1].item())
    mem_gen = torch.cuda.max_memory_allocated() / 1e9

    # condition-blind deterministic baseline: the standard VAE decode of the SAME latent
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        z = lat.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        img = pipe.vae.decode(z, return_dict=False)[0]
        img = pipe.image_processor.postprocess(img, output_type="pil")[0]
    torch.cuda.synchronize(); t_vae = time.time() - t0
    mem_vae = torch.cuda.max_memory_allocated() / 1e9
    img.save(os.path.join(out_vae, f"{idx}.png"))

    torch.save({"latent": lat.half().cpu(), "sigma": final_sigma,
                "xt24": xt.half().cpu(), "sigma24": captured["sigma"], "capture_step": a.capture_step,
                "caption": cap, "seed": a.seed, "steps": a.steps, "guidance": a.guidance, "t_gen_s": t_gen}, pt)
    with open(log_path, "a") as f:
        f.write(json.dumps({"idx": idx, "t_gen_s": t_gen, "t_vae_s": t_vae, "peak_mem_gen_gb": mem_gen,
                            "peak_mem_vae_gb": mem_vae, "sigma24": captured["sigma"]}) + "\n")
    if n % 10 == 0:
        print(f"[{n+1}/{len(ids)}] {idx} gen {t_gen:.2f}s vae {t_vae:.3f}s sigma24={captured['sigma']:.3f} mem {mem_gen:.1f}GB", flush=True)
print("done:", out_lat, out_vae)
