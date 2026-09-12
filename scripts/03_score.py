#!/usr/bin/env python
"""03: Score every decoder variant on dev-200 (the DEV_SETTING metrics), write per-image CSV + summary table.

Adherence (the signal):  Canny F1 between cv2.Canny(gray(output), 100, 200) and the input condition.
   - matched 512 view : output at 512 (VAE native; PiD via the INTER_AREA downsample) vs the 512 condition
   - native 2048      : 2048 output vs the condition resampled to 2048 with NEAREST (BENCHMARK v1.1 rule);
                        for the 512-native VAE row this is a REFERENCE computed on a BICUBIC x4 upsample of its output (the paper's
                        dagger convention for 512-native rows; the interpolation route to 2048, not a native output)
   TWO F1 variants (BENCHMARK v1.8, 2026-09-12 update of this loop): `canny_f1_*` = F1 with a matching tolerance of ONE CONDITION PIXEL
   (1 px at 512, 4 px at 2048; BSDS-style dilation) = the paper metric and THE GATE at both resolutions; `canny_f1s_*` = the strict
   pixel-exact F1 of the original loop, kept as the 512 anchor (its 2048 value mostly measures the 4-px-thick resampled reference).
Fidelity vs the real source image (512), computed on the WHOLE RGB image (not edges): LPIPS (alex, lower better), PSNR, SSIM (higher better) [pyiqa]
No-reference quality (higher better): MUSIQ on the matched 512 view (comparable across rows) and at native resolution [pyiqa]
Cost: per-image generation / decode latency and peak memory from the run logs.

Variants scored (dir under outputs/): omini_vae (512), omini_pid_512 + omini_pid (2048), omini_pid_et24_512 + omini_pid_et24.
"""
import argparse, glob, json, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch, pandas as pd
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=REPO)
ap.add_argument("--variants", default="omini_vae,omini_pid,omini_pid_et24,omini_pid_et16")
a = ap.parse_args()
dev = os.path.join(a.repo, "dev200"); outs = os.path.join(a.repo, "outputs"); res = os.path.join(a.repo, "results")
os.makedirs(res, exist_ok=True)
ids = sorted(json.load(open(os.path.join(dev, "captions.json"))))

import pyiqa
dev_t = "cuda" if torch.cuda.is_available() else "cpu"
M = {k: pyiqa.create_metric(k, device=dev_t) for k in ["lpips", "psnr", "ssim", "musiq"]}

def canny(img_rgb):                     # uint8 RGB -> binary edge map
    return cv2.Canny(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), 100, 200) > 0

def f1(pred, ref):                      # strict pixel-exact F1 (the original loop metric; the 512 anchor)
    tp = np.logical_and(pred, ref).sum(); fp = np.logical_and(pred, ~ref).sum(); fn = np.logical_and(~pred, ref).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))

def f1_tol(pred, ref, tol):             # BENCHMARK v1.8 paper metric: tolerance of one condition pixel (tol = res // 512)
    if tol <= 0:
        return f1(pred, ref)
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    ref_d = cv2.dilate(ref.astype(np.uint8), k) > 0; pred_d = cv2.dilate(pred.astype(np.uint8), k) > 0
    prec = np.logical_and(pred, ref_d).sum() / max(pred.sum(), 1); rec = np.logical_and(ref, pred_d).sum() / max(ref.sum(), 1)
    return float(2 * prec * rec / max(prec + rec, 1e-9))

def load(p): return np.array(Image.open(p).convert("RGB"))
def tens(arr): return torch.from_numpy(arr).permute(2, 0, 1)[None].float().div(255).to(dev_t)

def native_dir(v):
    return os.path.join(outs, v)

rows = []
for idx in ids:
    src = load(os.path.join(dev, "images", f"{idx}.png"))
    cond512 = load(os.path.join(dev, "canny", f"{idx}.png"))[..., 0] > 0
    for v in a.variants.split(","):
        nat_p = os.path.join(native_dir(v), f"{idx}.png")
        if not os.path.exists(nat_p):
            continue
        nat = load(nat_p); H = nat.shape[0]
        if H == 512:
            view512 = nat
        else:
            p512 = os.path.join(native_dir(v) + "_512", f"{idx}.png")
            view512 = load(p512) if os.path.exists(p512) else cv2.resize(nat, (512, 512), interpolation=cv2.INTER_AREA)
        r = {"idx": idx, "variant": v, "native_res": H}
        e512 = canny(view512)
        r["canny_f1_512"] = f1_tol(e512, cond512, 1); r["canny_f1s_512"] = f1(e512, cond512)
        HR = 2048
        cond_hr = cv2.resize(cond512.astype(np.uint8), (HR, HR), interpolation=cv2.INTER_NEAREST) > 0
        if H == HR:
            e_hr = canny(nat); r["native_via"] = "native"
        else:   # 512-native decoders (VAE): REFERENCE value on a BICUBIC x4 upsample (the paper's dagger convention, 2026-09-12; was bilinear); not a claim
            e_hr = canny(cv2.resize(nat, (HR, HR), interpolation=cv2.INTER_CUBIC)); r["native_via"] = "bicubic_x4"
        r["canny_f1_native"] = f1_tol(e_hr, cond_hr, HR // 512); r["canny_f1s_native"] = f1(e_hr, cond_hr)
        with torch.no_grad():
            r["lpips"] = float(M["lpips"](tens(view512), tens(src)))
            r["psnr"] = float(M["psnr"](tens(view512), tens(src)))
            r["ssim"] = float(M["ssim"](tens(view512), tens(src)))
            r["musiq_native"] = float(M["musiq"](tens(nat)))
            r["musiq_512"] = float(M["musiq"](tens(view512)))
        rows.append(r)

df = pd.DataFrame(rows); df.to_csv(os.path.join(res, "dev200_per_image.csv"), index=False)

# cost from logs
lat = {}
if os.path.exists(os.path.join(res, "log_01_generate.jsonl")):
    g = pd.read_json(os.path.join(res, "log_01_generate.jsonl"), lines=True)
    lat["gen_s"] = g.t_gen_s.mean(); lat["omini_vae"] = (g.t_vae_s.mean(), g.peak_mem_vae_gb.max())
if os.path.exists(os.path.join(res, "log_02_pid.jsonl")):
    p = pd.read_json(os.path.join(res, "log_02_pid.jsonl"), lines=True)
    for v, gdf in p.groupby("variant"):
        lat["omini_pid" + ("" if v == "final" else f"_{v}")] = (gdf.t_dec_s.mean(), gdf.peak_mem_gb.max())

agg = df.groupby("variant").agg(n=("idx", "count"), native_res=("native_res", "first"),
                                canny_f1_512=("canny_f1_512", "mean"), canny_f1_native=("canny_f1_native", "mean"),
                                canny_f1s_512=("canny_f1s_512", "mean"), canny_f1s_native=("canny_f1s_native", "mean"),
                                lpips=("lpips", "mean"), psnr=("psnr", "mean"), ssim=("ssim", "mean"),
                                musiq_native=("musiq_native", "mean"), musiq_512=("musiq_512", "mean")).reset_index()
agg["decode_s"] = agg.variant.map(lambda v: round(lat.get(v, (np.nan, np.nan))[0], 3))
agg["peak_mem_gb"] = agg.variant.map(lambda v: round(lat.get(v, (np.nan, np.nan))[1], 1))
order = {"omini_vae": 0, "omini_pid": 1, "omini_pid_et24": 2, "omini_pid_et16": 3}
agg = agg.sort_values("variant", key=lambda s: s.map(lambda v: order.get(v, 9))).reset_index(drop=True)
agg.to_csv(os.path.join(res, "dev200_summary.csv"), index=False)

names = {"omini_vae": "OminiControl + VAE decode (512, native)",
         "omini_pid": "OminiControl + vanilla PiD (final latent, 2048)",
         "omini_pid_et24": "OminiControl + vanilla PiD, early-terminated at 24/28 (2048)",
         "omini_pid_et16": "OminiControl + vanilla PiD, early-terminated at 16/28 (2048)"}
lines = ["| Decoder | n | Canny F1 @512 (matched, tolerant) ↑ | Canny F1 @2048 (native, tolerant; VAE row = bicubic x4 reference) ↑ | strict F1 @512 (anchor) ↑ | strict F1 @2048 (metric-dominated) | LPIPS ↓ | PSNR ↑ | SSIM ↑ | MUSIQ @512 (matched) ↑ | MUSIQ (native) ↑ | decode s/img ↓ | peak mem GB ↓ |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in agg.iterrows():
    f1n = f"{r.canny_f1_native:.4f}" + (" (bicubic x4 ref.)" if r.native_res == 512 else "")
    lines.append(f"| {names.get(r.variant, r.variant)} | {int(r.n)} | {r.canny_f1_512:.4f} | {f1n} | {r.canny_f1s_512:.4f} | {r.canny_f1s_native:.4f} | {r.lpips:.4f} | {r.psnr:.2f} | {r.ssim:.4f} | {r.musiq_512:.2f} | {r.musiq_native:.2f} | {r.decode_s} | {r.peak_mem_gb} |")
gen_line = f"\nGeneration (FLUX.1-dev + OminiControl canny LoRA, 28 steps @512, seed 0): {lat.get('gen_s', float('nan')):.2f} s/img, shared by every row.\n" if "gen_s" in lat else ""
open(os.path.join(res, "dev200_summary.md"), "w").write("\n".join(lines) + "\n" + gen_line)
print("\n".join(lines)); print(gen_line)
