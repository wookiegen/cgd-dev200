"""tab:efficiency: per-image latency and peak memory to reach an output at 512 / 2048 / 4096 on ONE GPU.

Rows measured (method-independent):
  FLUX + VAE     FLUX.1-dev generation at the OUTPUT resolution (28 steps, guidance 3.5) + VAE decode
  FLUX + PiD     FLUX.1-dev generation at ONE QUARTER of the output resolution (28 steps, and truncated at K = 16 / 24) + vanilla PiD
                 (4 steps): 512 -> 2048 with the 2k checkpoint, 1024 -> 4096 with the 2kto4k v1.5 checkpoint
Generation is plain FLUX.1-dev (no control adapter); a controller adds its own condition tokens on top, which is the same for every decoder.
Latency = wall clock per image after one warm-up image, median over --n images; memory = torch peak allocated (GB). Native FLUX at 4096 is
attempted once and marked impractical on OOM or if a single image exceeds --max-s seconds.
Usage: CUDA_VISIBLE_DEVICES=3 python measure_efficiency.py [--n 5] [--out ../results/bench/efficiency.json]
"""
import argparse
import csv
import json
import statistics
import time

import numpy as np
import torch

from common import REPO_BENCH

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5); ap.add_argument("--max-s", type=float, default=900)
ap.add_argument("--out", default=str(REPO_BENCH.parent / "results" / "bench" / "efficiency.json"))
ap.add_argument("--skip-4096-native", action="store_true")
a = ap.parse_args()
caps = [r["caption"] for r in list(csv.DictReader(open(REPO_BENCH / "multigen5k" / "manifest.csv")))[: a.n + 1]]
dev = "cuda"
results = []


def timed(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    out = fn(); torch.cuda.synchronize()
    return out, time.time() - t0, torch.cuda.max_memory_allocated() / 1e9


def record(**kw):
    kw["gpu"] = torch.cuda.get_device_name(0); results.append(kw); print(json.dumps(kw), flush=True)
    json.dump(results, open(a.out, "w"), indent=1)


from diffusers import FluxPipeline  # noqa: E402
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(dev)
pipe.set_progress_bar_config(disable=True)


def gen(res, steps, cap, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    return pipe(prompt=cap, height=res, width=res, num_inference_steps=steps, guidance_scale=3.5, generator=g, output_type="latent").images


def vae_decode(packed, res):
    lat = FluxPipeline._unpack_latents(packed, res, res, pipe.vae_scale_factor)
    z = lat.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    return pipe.vae.decode(z, return_dict=False)[0]


# ---- FLUX generation at several resolutions / step counts (+ VAE decode)
plan = [(512, 28), (512, 24), (512, 16), (1024, 28), (1024, 24), (1024, 16), (2048, 28)] + ([] if a.skip_4096_native else [(4096, 28)])
lat_cache = {}
for res, steps in plan:
    try:
        with torch.no_grad():
            _ = gen(res, 2 if res >= 4096 else steps, caps[0])                      # warm-up (short at 4096)
            tg, mg, tv, mv = [], [], [], []
            for i in range(1, a.n + 1 if res < 4096 else 2):
                (packed, t, m) = timed(lambda: gen(res, steps, caps[i]))
                tg.append(t); mg.append(m)
                if t > a.max_s:
                    raise TimeoutError(f"{t:.0f}s per image")
                (_, t2, m2) = timed(lambda: vae_decode(packed, res))
                tv.append(t2); mv.append(m2)
                if steps == 28 and i == 1:
                    lat_cache[res] = FluxPipeline._unpack_latents(packed, res, res, pipe.vae_scale_factor).half().cpu()
        record(pipeline="FLUX + VAE", output=res, gen_res=res, gen_steps=steps, dec_steps=1, gen_s=statistics.median(tg), vae_s=statistics.median(tv),
               latency_s=statistics.median(tg) + statistics.median(tv), peak_mem_gb=max(mg + mv), n=len(tg))
    except (torch.cuda.OutOfMemoryError, TimeoutError, RuntimeError) as e:  # noqa: PERF203
        record(pipeline="FLUX + VAE", output=res, gen_res=res, gen_steps=steps, dec_steps=1, latency_s=None, peak_mem_gb=None, n=0, note=f"impractical: {type(e).__name__}: {str(e)[:80]}")
        torch.cuda.empty_cache()

# ---- generation cost at the quarter resolution for the truncated settings is already in `plan` (512 / 1024 at 28, 24, 16)
del pipe; torch.cuda.empty_cache()

# ---- PiD decode: 512 -> 2048 (2k ckpt), 1024 -> 4096 (2kto4k v1.5 ckpt)
from flux_pid import PiD  # noqa: E402
for ckpt, gres, out_res in [("2k", 512, 2048), ("2kto4k_v1pt5", 1024, 4096)]:
    try:
        pid = PiD(ckpt_type=ckpt)
        lat = lat_cache.get(gres)
        if lat is None:
            lat = torch.randn(1, 16, gres // 8, gres // 8).half()
        with torch.no_grad():
            _ = pid.decode(lat, [caps[0]], steps=4)
            ts, ms = [], []
            for i in range(1, a.n + 1):
                (_, t, m) = timed(lambda: pid.decode(lat, [caps[i]], steps=4)); ts.append(t); ms.append(m)
        gen_rows = {r["gen_steps"]: r for r in results if r["pipeline"] == "FLUX + VAE" and r["gen_res"] == gres and r.get("gen_s")}
        for K in [28, 24, 16]:
            g = gen_rows.get(K)
            record(pipeline="FLUX + PiD", output=out_res, gen_res=gres, gen_steps=K, dec_steps=4, pid_ckpt=pid.ckpt.experiment,
                   gen_s=g["gen_s"] if g else None, dec_s=statistics.median(ts), latency_s=(g["gen_s"] if g else float("nan")) + statistics.median(ts),
                   peak_mem_gb=max([g["peak_mem_gb"]] if g else [0] + ms), dec_peak_mem_gb=max(ms), n=len(ts))
        del pid; torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        record(pipeline="FLUX + PiD", output=out_res, gen_res=gres, gen_steps=None, dec_steps=4, latency_s=None, peak_mem_gb=None, n=0, note=f"failed: {type(e).__name__}: {str(e)[:100]}")
print("wrote", a.out)
