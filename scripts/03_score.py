#!/usr/bin/env python
"""03: Score every decoder row on dev-200 under the FINAL protocol (BENCHMARK v1.11 / update.md) and write the reference table.

Every row gets a 512 column and a 2048 column, both in the paper's tolerant canny F1 (one condition pixel: 1 px at 512, 4 px at 2048):
  512  = the matched view: the output at 512 (VAE native; 2048 outputs via the INTER_AREA downsample, the pinned view)
  2048 = the native output when the row produces 2048; for 512-native rows (VAE decode, Real, VAE round trip) it is the INTERPOLATION
         route, bicubic x4 of the 512 output, marked with a dagger (‡) = "post-enlarge"
  2048 vs native condition = F1 at 2048 against the 2048 edge map (Canny of the PiD round trip of the real image; tolerance 1 px), the
         native-condition setting; the dagger applies as above
  strict@512 = the pixel-exact F1 of the original loop (anchor)
Rows (dirs under outputs/, present ones are scored): omini_vae (512 native), omini_vae_gen2048 (OminiControl GENERATING at 2048 = the
native route), omini_pid / _et24 / _et16 (4-step distilled student), omini_pidt / _et24 / _et16 (undistilled teacher, 25 steps CFG 5);
reference rows: real (dev200/images), vae_roundtrip / pid_roundtrip (mapped from the paper benchmark's outputs by sample id).
Also: LPIPS / PSNR / SSIM vs the source at 512, MUSIQ at 512 and native, decode s/img and peak memory from the run logs.
"""
import argparse, csv, glob, json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "bench"))
import numpy as np, cv2, torch, pandas as pd
from PIL import Image
from scorers import CannyF1  # the paper's scorer (tolerant + strict), bench/scorers.py

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=REPO)
ap.add_argument("--bench-root", default=os.environ.get("CGD_BENCH_ROOT", "/shared4/Project_Archive/2026-conditional-decoding/bench"),
                help="materialized paper benchmark (round trips, canny2048); defaults to the team share, override with CGD_BENCH_ROOT=/data/wookiekim/cgd/data/bench on the machine that holds the local copy")
ap.add_argument("--variants", default="omini_vae,omini_vae_gen2048,omini_pid,omini_pid_et24,omini_pid_et16,omini_pidt,omini_pidt_et24,omini_pidt_et16")
a = ap.parse_args()
dev = os.path.join(a.repo, "dev200"); outs = os.path.join(a.repo, "outputs"); res = os.path.join(a.repo, "results"); os.makedirs(res, exist_ok=True)
ids = sorted(json.load(open(os.path.join(dev, "captions.json"))))
# dev idx -> paper sample_id (the dev-200 is a subset of multigen5k)
sid_of = {r["dev200_idx"].zfill(3): r["sample_id"] for r in csv.DictReader(open(os.path.join(a.repo, "bench", "multigen5k", "manifest.csv"))) if r["dev200_idx"]}
B = a.bench_root
HR = 2048
sc = CannyF1()
import pyiqa
dev_t = "cuda" if torch.cuda.is_available() else "cpu"
M = {k: pyiqa.create_metric(k, device=dev_t) for k in ["lpips", "psnr", "ssim", "musiq"]}


def load(p): return np.array(Image.open(p).convert("RGB"))
def tens(arr): return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].float().div(255).to(dev_t)
def up4(img512): return cv2.resize(img512, (HR, HR), interpolation=cv2.INTER_CUBIC)


def native_cond(idx):
    p = os.path.join(B, "multigen5k", "conditions", "canny2048", f"{sid_of[idx]}.png") if idx in sid_of else None
    return load(p) if p and os.path.exists(p) else None


def ref_paths(row, idx):
    """512 image and (optional) 2048 image of a reference row."""
    sid = sid_of.get(idx)
    if row == "real":
        return os.path.join(dev, "images", f"{idx}.png"), None
    if row == "vae_roundtrip":
        return os.path.join(B, "outputs", "ref", "vae_roundtrip", "multigen5k", f"{sid}.png"), None
    if row == "pid_roundtrip":
        return os.path.join(B, "outputs", "ref", "pid_roundtrip_512", "multigen5k", f"{sid}.png"), os.path.join(B, "outputs", "ref", "pid_roundtrip", "multigen5k", f"{sid}.png")
    if row == "pidt_roundtrip":
        return os.path.join(B, "outputs", "ref", "pidt_roundtrip_512", "multigen5k", f"{sid}.png"), os.path.join(B, "outputs", "ref", "pidt_roundtrip", "multigen5k", f"{sid}.png")
    raise KeyError(row)


def score_pair(view512, nat, cond512, cond2048, src, dagger):
    """view512: 512 view; nat: 2048 image (native or bicubic upsample when dagger)."""
    r = {"canny_f1_2048_natcond": np.nan}   # seeded so the column always exists, even where no 2048 condition is available
    s = sc.score(view512, cond512); r["canny_f1_512"] = s["f1"]; r["canny_f1s_512"] = s["f1_strict"]
    s = sc.score(nat, cond512); r["canny_f1_2048"] = s["f1"]; r["canny_f1s_2048"] = s["f1_strict"]
    if cond2048 is not None:
        r["canny_f1_2048_natcond"] = sc.score(nat, cond2048)["f1"]
    r["dagger"] = int(dagger)
    with torch.no_grad():
        r["lpips"] = float(M["lpips"](tens(view512), tens(src))); r["psnr"] = float(M["psnr"](tens(view512), tens(src))); r["ssim"] = float(M["ssim"](tens(view512), tens(src)))
        r["musiq_512"] = float(M["musiq"](tens(view512))); r["musiq_native"] = float(M["musiq"](tens(nat)))
    return r


missing_ref = []
rows = []
for idx in ids:
    src = load(os.path.join(dev, "images", f"{idx}.png"))
    cond512 = load(os.path.join(dev, "canny", f"{idx}.png")); cond2048 = native_cond(idx)
    # reference rows
    for ref in ["real", "vae_roundtrip", "pid_roundtrip", "pidt_roundtrip"]:
        p512, p2048 = ref_paths(ref, idx)
        if not os.path.exists(p512):
            missing_ref.append(ref)
            continue
        v = load(p512)
        if p2048 and os.path.exists(p2048):
            rows.append({"idx": idx, "variant": ref, "native_res": HR, **score_pair(v, load(p2048), cond512, cond2048, src, False)})
        else:
            rows.append({"idx": idx, "variant": ref, "native_res": 512, **score_pair(v, up4(v), cond512, cond2048, src, True)})
    # decoder rows
    for v in a.variants.split(","):
        nat_p = os.path.join(outs, v, f"{idx}.png")
        if not os.path.exists(nat_p):
            continue
        nat = load(nat_p); H = nat.shape[0]
        if H == 512:
            rows.append({"idx": idx, "variant": v, "native_res": 512, **score_pair(nat, up4(nat), cond512, cond2048, src, True)})
        else:
            p512 = os.path.join(outs, v + "_512", f"{idx}.png")
            view512 = load(p512) if os.path.exists(p512) else cv2.resize(nat, (512, 512), interpolation=cv2.INTER_AREA)
            rows.append({"idx": idx, "variant": v, "native_res": H, **score_pair(view512, nat, cond512, cond2048, src, False)})

df = pd.DataFrame(rows); df.to_csv(os.path.join(res, "dev200_per_image.csv"), index=False)

# cost from logs
lat = {}
for f, key in [("log_01_generate.jsonl", None), ("log_02_pid.jsonl", "pid"), ("log_02b_teacher.jsonl", "pidt"), ("log_01b_gen2048.jsonl", "gen2048")]:
    p = os.path.join(res, f)
    if not os.path.exists(p):
        continue
    g = pd.read_json(p, lines=True)
    if key is None:
        lat["gen_s"] = g.t_gen_s.mean(); lat["omini_vae"] = (g.t_vae_s.mean(), g.peak_mem_vae_gb.max())
    elif key == "gen2048":
        lat["omini_vae_gen2048"] = (g.t_gen_vae_s.mean(), g.peak_mem_gb.max())
    else:
        for v, gdf in g.groupby("variant"):
            lat[f"omini_{key}" + ("" if v == "final" else f"_{v}")] = (gdf.t_dec_s.mean(), gdf.peak_mem_gb.max())

if missing_ref:
    miss = sorted(set(missing_ref))
    print(f"\nNOTE: no images found for the reference row(s) {miss} under CGD_BENCH_ROOT={B}.\n"
          f"      Those rows are produced on the group server and are not part of the public release, so they are omitted\n"
          f"      from the table below; every decoder row you generated yourself is still scored. Set CGD_BENCH_ROOT to a\n"
          f"      tree that contains outputs/ref/ if you need them.\n", flush=True)
if df.empty:
    raise SystemExit(f"no rows scored: neither reference images under CGD_BENCH_ROOT={B} nor decoder outputs under {outs} were found")

agg = df.groupby("variant").agg(n=("idx", "count"), native_res=("native_res", "first"), dagger=("dagger", "first"),
                                canny_f1_512=("canny_f1_512", "mean"), canny_f1_2048=("canny_f1_2048", "mean"),
                                canny_f1_2048_natcond=("canny_f1_2048_natcond", "mean"), canny_f1s_512=("canny_f1s_512", "mean"),
                                lpips=("lpips", "mean"), psnr=("psnr", "mean"), ssim=("ssim", "mean"),
                                musiq_512=("musiq_512", "mean"), musiq_native=("musiq_native", "mean")).reset_index()
agg["decode_s"] = agg.variant.map(lambda v: round(lat.get(v, (np.nan, np.nan))[0], 2)); agg["peak_mem_gb"] = agg.variant.map(lambda v: round(lat.get(v, (np.nan, np.nan))[1], 1))
order = {"real": 0, "vae_roundtrip": 1, "pid_roundtrip": 2, "pidt_roundtrip": 3, "omini_vae": 10, "omini_vae_gen2048": 11, "omini_pid": 20, "omini_pid_et24": 21, "omini_pid_et16": 22,
         "omini_pidt": 30, "omini_pidt_et24": 31, "omini_pidt_et16": 32}
agg = agg.sort_values("variant", key=lambda s: s.map(lambda v: order.get(v, 99))).reset_index(drop=True)
agg.to_csv(os.path.join(res, "dev200_summary.csv"), index=False)

names = {"real": "Real image (512; 2048 = bicubic x4 ‡)", "vae_roundtrip": "VAE round trip (decode ceiling; 2048 = bicubic x4 ‡)",
         "pid_roundtrip": "PiD round trip, student (generative ceiling; 2048 native)", "pidt_roundtrip": "PiD round trip, teacher",
         "omini_vae": "OminiControl + VAE decode (512 native; 2048 = bicubic x4 ‡, the interpolation route)",
         "omini_vae_gen2048": "OminiControl GENERATING at 2048 + VAE decode (the native route)",
         "omini_pid": "OminiControl + vanilla PiD student, final latent (28/28)", "omini_pid_et24": "OminiControl + vanilla PiD student, K=24", "omini_pid_et16": "OminiControl + vanilla PiD student, K=16 **(gate row)**",
         "omini_pidt": "OminiControl + vanilla PiD teacher, final latent (28/28)", "omini_pidt_et24": "OminiControl + vanilla PiD teacher, K=24", "omini_pidt_et16": "OminiControl + vanilla PiD teacher, K=16 **(gate row for teacher-based CGD)**"}


def f(v, nd=4, dag=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.{nd}f}" + ("‡" if dag else "")


lines = ["| Decoder | n | Canny F1 @512 vs c512 (1 px) ↑ | Canny F1 @2048 vs c512 (4 px) ↑ | Canny F1 @2048 vs c2048 (1 px) ↑ | strict F1 @512 (anchor) | LPIPS ↓ | PSNR ↑ | SSIM ↑ | MUSIQ @512 ↑ | MUSIQ @2048 ↑ | s/img ↓ | peak GB ↓ |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in agg.iterrows():
    dag = bool(r.dagger)
    lines.append(f"| {names.get(r.variant, r.variant)} | {int(r.n)} | {f(r.canny_f1_512)} | {f(r.canny_f1_2048, dag=dag)} | {f(r.canny_f1_2048_natcond, dag=dag)} | {f(r.canny_f1s_512)} | "
                 f"{f(r.lpips)} | {f(r.psnr, 2)} | {f(r.ssim)} | {f(r.musiq_512, 2)} | {f(r.musiq_native, 2)} | {f(r.decode_s, 2)} | {f(r.peak_mem_gb, 1)} |")
gen_line = f"\nGeneration at 512 (FLUX.1-dev + OminiControl canny LoRA, 28 steps, seed 0): {lat.get('gen_s', float('nan')):.2f} s/img, shared by every row except the native-route row, whose s/img includes its own 2048 generation. ‡ = bicubic x4 upsample of a 512 output (interpolation route, not a native output). Native condition = Canny of the PiD round trip at 2048, decoded at the HELD-OUT decoder seed 7 (`bench/build_native_conditions.py --src-name pid_roundtrip_s7`; BENCHMARK v1.16). No evaluated row uses seed 7, so no row shares sampler noise with its own reference; the seed-0 round-trip row therefore scores about 0.71 there, not 1.0, and that value is the sampler-noise floor of the column.\n" if "gen_s" in lat else ""
open(os.path.join(res, "dev200_summary.md"), "w").write("\n".join(lines) + "\n" + gen_line)
print("\n".join(lines)); print(gen_line)
