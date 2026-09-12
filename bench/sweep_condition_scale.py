"""Condition-scale sweep for fig:ceiling (BENCHMARK v1 Section 8, fig:ceiling spec) on the canny dev-200 under OminiControl.

x = OminiControl `condition_scale`: an additive bias of log(scale) on the condition-to-image attention logits (1.0 = the released
    setting and no bias; 0 = the condition tokens are fully suppressed, i.e. text-only generation; > 1 = stronger control).
y = canny F1 (tolerant, v1.8; strict kept) of the VAE decode at 512 and of vanilla PiD at K = 28 and K = 24, at the 512 view and native 2048.
Ceiling = the VAE round trip of the same 200 real images (results/bench/multigen5k/vae_roundtrip@512_per_image.csv), PiD round trip likewise.

Outputs per scale s under $CGD_BENCH_ROOT/sweep/condition_scale/s<scale>/: latents/<sid>.pt (x_0 + xt24 + sigma24), vae@28/<sid>.png (512),
pid@<K>/<sid>.png (2048), pid@<K>_512/<sid>.png (INTER_AREA view). `--assemble` scores everything on the CPU and writes
results/bench/sweep/condition_scale.json (+ _per_image.csv, SUMMARY.md).

Usage (inside the container, one process per GPU, sharded over scales):
  CUDA_VISIBLE_DEVICES=0 python sweep_condition_scale.py --scales 0,0.25,0.5,0.75,1,1.5,2,4 --shard 0 --nshards 2
  CUDA_VISIBLE_DEVICES=1 python sweep_condition_scale.py --scales 0,0.25,0.5,0.75,1,1.5,2,4 --shard 1 --nshards 2
  python sweep_condition_scale.py --scales 0,0.25,0.5,0.75,1,1.5,2,4 --assemble
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

from common import OUT_ROOT, REPO_BENCH

ap = argparse.ArgumentParser()
ap.add_argument("--scales", default="0,0.25,0.5,0.75,1,1.5,2,4")
ap.add_argument("--ks", default="28,24")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--steps", type=int, default=28); ap.add_argument("--guidance", type=float, default=3.5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--omini", default=os.environ.get("OMINI_ROOT", str(OUT_ROOT.parent.parent / "OminiControl")))
ap.add_argument("--assemble", action="store_true", help="score the finished outputs (CPU) and write results/bench/sweep/")
a = ap.parse_args()

SCALES = [float(s) for s in a.scales.split(",")]
KS = [int(k) for k in a.ks.split(",")]
SWEEP = OUT_ROOT / "sweep" / "condition_scale"
RES = REPO_BENCH.parent / "results" / "bench" / "sweep"
COND = OUT_ROOT / "multigen5k" / "conditions" / "canny"


def tag(s: float) -> str:
    return f"s{s:g}"


def dev200_rows():
    """The 200 dev images = the first multigen5k row of each dev200_idx (two dev images are byte-identical duplicates in the 5k set)."""
    rows = [r for r in csv.DictReader(open(REPO_BENCH / "multigen5k" / "manifest.csv")) if r["dev200_idx"]]
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: (int(r["dev200_idx"]), r["sample_id"])):
        if r["dev200_idx"] not in seen:
            seen.add(r["dev200_idx"]); out.append(r)
    return out[: a.limit or None]


rows = dev200_rows()

# ------------------------------------------------------------------ assemble (CPU)
if a.assemble:
    from scorers import CannyF1
    sc = CannyF1(); RES.mkdir(parents=True, exist_ok=True)
    conds = {r["sample_id"]: np.array(__import__("PIL.Image", fromlist=["Image"]).open(COND / f"{r['sample_id']}.png").convert("RGB")) for r in rows}
    per, means = [], {}
    for s in SCALES:
        d = SWEEP / tag(s)
        views = {"vae@512": (d / "vae@28", 512)}
        for K in KS:
            views[f"pid_k{K}@512"] = (d / f"pid@{K}_512", 512); views[f"pid_k{K}@2048"] = (d / f"pid@{K}", 2048)
        acc = {k: {"f1": [], "f1_strict": []} for k in views}
        n_ok = 0
        for r in rows:
            sid = r["sample_id"]; rec = {"scale": s, "sample_id": sid}; ok = True
            for k, (vd, res) in views.items():
                p = vd / f"{sid}.png"
                if not p.exists():
                    ok = False; continue
                img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                assert img.shape[0] == res, (p, img.shape)
                out = sc.score(img, conds[sid]); rec[f"{k}_f1"] = out["f1"]; rec[f"{k}_f1_strict"] = out["f1_strict"]
                acc[k]["f1"].append(out["f1"]); acc[k]["f1_strict"].append(out["f1_strict"])
            n_ok += ok; per.append(rec)
        means[tag(s)] = {"scale": s, "n_complete": n_ok, **{f"{k}_{m}": float(np.mean(v)) for k, mv in acc.items() for m, v in mv.items() if v}}
        print(tag(s), n_ok, {k: round(v, 4) for k, v in means[tag(s)].items() if k.endswith("_f1")}, flush=True)
    # decoder-only ceilings on the same 200 images (round trips of the REAL images; harness records on the full 5k)
    ids = {r["sample_id"] for r in rows}; ceil = {}
    for name, f in [("vae_roundtrip@512", "vae_roundtrip@512_per_image.csv"), ("pid_roundtrip@512", "pid_roundtrip@512_per_image.csv"),
                    ("pid_roundtrip@2048", "pid_roundtrip@2048_per_image.csv")]:
        p = REPO_BENCH.parent / "results" / "bench" / "multigen5k" / f
        if p.exists():
            rr = [x for x in csv.DictReader(open(p)) if x["sample_id"] in ids]
            ceil[name] = {"f1": float(np.mean([float(x["canny_f1"]) for x in rr])), "f1_strict": float(np.mean([float(x["canny_f1_strict"]) for x in rr])), "n": len(rr)}
    out = {"spec": "fig:ceiling, BENCHMARK v1 Section 8; canny dev-200 (first multigen5k row per dev200_idx), OminiControl canny LoRA, seed 0, 28 steps, guidance 3.5",
           "x": "OminiControl condition_scale (log-scale attention bias; 1 = released setting, 0 = condition suppressed)", "scales": SCALES, "ks": KS,
           "metric": "canny F1 tolerant (v1.8; one condition pixel), f1_strict = pixel-exact", "per_scale": means, "ceilings_dev200": ceil}
    json.dump(out, open(RES / "condition_scale.json", "w"), indent=1)
    keys = sorted({k for x in per for k in x}, key=lambda k: (k not in ("scale", "sample_id"), k))
    with open(RES / "condition_scale_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(per)
    cols = ["vae@512"] + [f"pid_k{K}@{r}" for K in KS for r in (512, 2048)]
    with open(RES / "SUMMARY.md", "w") as f:
        f.write("# Condition-scale sweep (fig:ceiling), canny dev-200, OminiControl, tolerant canny F1\n\n| scale | n | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols) + "\n")
        for s in SCALES:
            m = means[tag(s)]; f.write(f"| {s:g} | {m['n_complete']} | " + " | ".join(f"{m.get(c + '_f1', float('nan')):.3f}" for c in cols) + " |\n")
        f.write("\nCeilings on the same 200 real images (round trips): " + ", ".join(f"{k} {v['f1']:.3f}" for k, v in ceil.items()) + "\n")
    print("wrote", RES / "condition_scale.json"); sys.exit(0)

# ------------------------------------------------------------------ generate + decode (GPU)
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from diffusers.pipelines import FluxPipeline  # noqa: E402

my_scales = [s for i, s in enumerate(SCALES) if i % a.nshards == a.shard]
dev = "cuda"; H = W = 512; VSF = 8
sys.path.insert(0, a.omini)
from omini.pipeline.flux_omini import Condition, generate  # noqa: E402
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(dev)
pipe.load_lora_weights("Yuanshi/OminiControl", weight_name="experimental/canny.safetensors", adapter_name="canny"); pipe.set_adapters(["canny"])
pipe.set_progress_bar_config(disable=True)
from flux_pid import PiD  # noqa: E402  (chdir's into the PiD root; every path below is absolute)
pid = PiD(ckpt_type="2k")
print(f"sweep shard {a.shard}/{a.nshards}: scales {my_scales}, {len(rows)} images, K in {KS}", flush=True)

for s in my_scales:
    d = SWEEP / tag(s); d_lat = d / "latents"; d_vae = d / "vae@28"
    for x in [d_lat, d_vae] + [d / f"pid@{K}" for K in KS] + [d / f"pid@{K}_512" for K in KS]:
        x.mkdir(parents=True, exist_ok=True)
    log = d / "log.jsonl"; t0 = time.time(); n = 0
    for r in rows:
        sid = r["sample_id"]
        if (d_vae / f"{sid}.png").exists() and all((d / f"pid@{K}" / f"{sid}.png").exists() for K in KS):
            continue
        cond = Image.open(COND / f"{sid}.png").convert("RGB")
        g = torch.Generator(device=dev).manual_seed(a.seed); captured = {}

        def cb(pipeline, i, t, kw):
            if (i + 1) in KS and (i + 1) != a.steps:
                captured[i + 1] = (kw["latents"].detach().clone(), float(pipeline.scheduler.sigmas[i + 1].item()))
            return kw

        t1 = time.time()
        with torch.no_grad():
            packed = generate(pipe, prompt=r["caption"], conditions=[Condition(cond, "canny")], condition_scale=s, height=H, width=W,
                              num_inference_steps=a.steps, guidance_scale=a.guidance, generator=g, output_type="latent", callback_on_step_end=cb).images
            lat = FluxPipeline._unpack_latents(packed, H, W, VSF)
            z = lat.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            img = pipe.image_processor.postprocess(pipe.vae.decode(z, return_dict=False)[0], output_type="pil")[0]
        torch.cuda.synchronize(); t_gen = time.time() - t1
        img.save(d_vae / f"{sid}.png")
        rec = {"latent": lat.half().cpu(), "sigma": 0.0, "caption": r["caption"], "seed": a.seed, "steps": a.steps, "guidance": a.guidance,
               "controller": "omini", "condition": "canny", "condition_scale": s, "sample_id": sid, "t_gen_s": t_gen}
        for K, (x, sg) in captured.items():
            rec[f"xt{K}"] = FluxPipeline._unpack_latents(x, H, W, VSF).half().cpu(); rec[f"sigma{K}"] = sg
        torch.save(rec, d_lat / f"{sid}.pt")
        for K in KS:
            lk, sg = (rec["latent"], 0.0) if K == a.steps else (rec[f"xt{K}"], float(rec[f"sigma{K}"]))
            y = pid.decode(lk, [r["caption"]], sigma=sg, steps=4, seed=a.seed)[0]
            cv2.imwrite(str(d / f"pid@{K}" / f"{sid}.png"), cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(d / f"pid@{K}_512" / f"{sid}.png"), cv2.cvtColor(cv2.resize(y, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        with open(log, "a") as f:
            f.write(json.dumps({"sample_id": sid, "condition_scale": s, "t_gen_s": t_gen, "sigmas": {K: sg for K, (_, sg) in captured.items()}}) + "\n")
        n += 1
        if n % 50 == 0:
            print(f"  {tag(s)} [{n}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)
    print(f"done {tag(s)}: {n} new in {time.time()-t0:.0f}s", flush=True)
print("shard done")
