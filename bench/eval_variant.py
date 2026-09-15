"""
FIXED 2026-09-15 (was: seg pairing broken and the gate failing OPEN). Three defects, all closed: the per-image filter tested
for the condition in the filename, which the harness only writes when a split has more than one condition, so seg records
never matched; the record-mean fallback derived the key "miou_img", which the harness does not write; and a criterion that
could not be evaluated printed "n/a" and dropped out of the verdict, so a seg variant with NO adherence data was judged by
LPIPS alone and printed GATE: PASS. Missing criteria now FAIL. The gate is K = 16 (BENCHMARK v1.17), now the default.
The adherence comparisons are strict > at both resolutions, so a tie FAILS.
One-command evaluation of a decoder VARIANT under the paper protocol (BENCHMARK v1.11), with the verdict against BASELINES.md.

Give it a directory of 2048x2048 PNGs named <sample_id>.png (one per manifest row, or per subset500 row) that decode the cached latents
`latents/<controller>/<condition>/<sid>.pt` at truncation point K. It runs the harness at the matched 512 view (adherence, no-ref, recon,
FID), at native 2048 (adherence, no-ref), and, for canny, at 2048 against the NATIVE 2048 condition; then prints the deltas and paired
bootstrap 95% CIs against the VAE decode and vanilla PiD (student; teacher if scored) at the same K, and a PASS / FAIL per gate criterion.

  CUDA_VISIBLE_DEVICES=g python eval_variant.py --name mycgd_v1 --gen-dir /path/to/2048pngs --controller omini --condition canny --k 16 [--subset500] [--gen-dir-512 <dir>] [--skip-noref]

Records land in results/variants/<name>/ (never in results/bench/, which holds the baselines). Reserved names: vae, pid_k*, pidt_k*, real, *_roundtrip, condvae.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import REPO_BENCH

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--gen-dir", required=True); ap.add_argument("--gen-dir-512", default=None)
ap.add_argument("--controller", default="omini"); ap.add_argument("--condition", default="canny", choices=["canny", "depth", "seg", "subject"])
ap.add_argument("--k", type=int, default=16, help="truncation point of the latents you decoded (BENCHMARK v1.17: the operating point and the gate are K = 16; 28 = full latent)")
ap.add_argument("--subset500", action="store_true"); ap.add_argument("--skip-noref", action="store_true"); ap.add_argument("--teacher", action="store_true", help="also compare with the teacher rows")
ap.add_argument("--n-boot", type=int, default=5000)
ap.add_argument("--report-only", action="store_true", help="skip the harness runs and rebuild the report from existing records")
a = ap.parse_args()
RESERVED = {"vae", "real", "condvae"}
assert a.name not in RESERVED and not a.name.startswith(("pid_k", "pidt_k")) and not a.name.endswith("_roundtrip"), "reserved method name"
split = {"canny": "multigen5k", "depth": "multigen5k", "seg": "ade20k_val2k", "subject": "dreambench750"}[a.condition]
# harness.py tags the filename with the condition ONLY when the split carries more than one (it runs them together otherwise),
# so testing for ".seg" / ".subject" in a filename can never match: those splits have exactly one condition. (2026-09-15)
SPLIT_CONDS = {"multigen5k": ["canny", "depth"], "ade20k_val2k": ["seg"], "coco_val5k": ["bbox"], "dreambench750": ["subject"], "div8k1k": ["canny"]}
COND_IN_NAME = len(SPLIT_CONDS[split]) > 1
ROOT = REPO_BENCH.parent; OUT = ROOT / "results" / "variants" / a.name; OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / "results" / "bench"
PY = sys.executable


def harness(res, metrics=None, cond_res=None):
    cmd = [PY, str(REPO_BENCH / "harness.py"), "--method", a.name, "--controller", a.controller, "--split", split, "--condition", a.condition,
           "--gen-dir", a.gen_dir, "--res", str(res), "--out-dir", str(OUT)]
    if a.gen_dir_512 and res == 512:
        cmd += ["--gen-dir-512", a.gen_dir_512]
    if metrics:
        cmd += ["--metrics", metrics]
    if cond_res:
        cmd += ["--cond-res", str(cond_res)]
    if a.subset500:
        cmd += ["--subset500"]
    print("+", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)


if not a.report_only:
    m512 = "adherence,recon,fid" if a.skip_noref else "adherence,noref,recon,fid"
    harness(512, m512)
    harness(2048, "adherence" if a.skip_noref else "adherence,noref")
    if a.condition == "canny":
        harness(2048, "adherence", cond_res=2048)


def load_records(d):
    out = {}
    for f in glob.glob(str(d / split / "*.json")):
        name = Path(f).name
        for rec in json.load(open(f)):
            cond_res = int(rec.get("adherence", {}).get("cond_res", 2048 if ".cond2048" in name else 512))
            sub = rec.get("subset") or ("subset500" if ".subset500" in name else "")
            via = "ref" if (rec.get("via") or "@2048ref" in name) else ""
            key = (rec["condition"], rec.get("controller", ""), rec["method"], rec["res"], cond_res, sub, via)
            e = out.setdefault(key, {"adherence": {}, "quality": {}, "n": rec.get("n"), "files": []})
            e["quality"].update(rec.get("quality", {})); e["adherence"] = e["adherence"] or dict(rec.get("adherence", {})); e["files"].append(f)
    return out


V = load_records(OUT); B = load_records(BASE)
sub = "subset500" if a.subset500 else ""


def pick(recs, method, res, cond_res=512, allow_full=True):
    r = recs.get((a.condition, a.controller, method, res, cond_res, sub, ""))
    if r is None and allow_full and sub:
        r = recs.get((a.condition, a.controller, method, res, cond_res, "", ""))
    return r


KEYCOL = {"canny": "canny_f1", "depth": "depth_rmse", "seg": "seg_miou_img", "subject": "subject_dino"}[a.condition]
# per-image CSV column -> the key the harness writes into the record's adherence dict. Deriving this by splitting on "_"
# gave "miou_img" for seg, which is not a key of that dict, so the seg fallback always returned None. (2026-09-15)
RECKEY = {"canny_f1": "f1", "depth_rmse": "rmse", "seg_miou_img": "miou", "subject_dino": "dino"}


def per_image(recs_dir, method, res, cond_res=512, subset=sub, col=None):
    """Per-image values of one column from the harness CSV of (method, res, condition, cond_res, subset); falls back to the full-set file."""
    col = col or KEYCOL
    pat = f"{a.controller}.{method}@{res}*_per_image.csv"
    fs = [f for f in glob.glob(str(recs_dir / split / pat))
          if (".cond2048" in f) == (cond_res == 2048) and ((".subset500" in f) == bool(subset)) and ".vlm" not in f
          and (f".{a.condition}" in Path(f).name if COND_IN_NAME else True)]
    if not fs and subset:
        return per_image(recs_dir, method, res, cond_res, subset="", col=col)
    if not fs:
        return {}
    return {r["sample_id"]: float(r[col]) for r in csv.DictReader(open(fs[0])) if r.get(col) not in (None, "", "nan")}


def paired(method_b, res, cond_res=512, col=None):
    """(variant mean, baseline mean, n) over the SAME images; None if either side lacks per-image data."""
    v = per_image(OUT, a.name, res, cond_res, col=col); b = per_image(BASE, method_b, res, cond_res, col=col)
    ids = sorted(set(v) & set(b))
    if len(ids) < 20:
        return None
    return float(np.mean([v[i] for i in ids])), float(np.mean([b[i] for i in ids])), len(ids)


def ci(v, b):
    ids = sorted(set(v) & set(b))
    if len(ids) < 20:
        return None
    d = np.array([v[i] - b[i] for i in ids]); rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), size=(a.n_boot, len(d)))].mean(axis=1)
    return d.mean(), *np.percentile(boots, [2.5, 97.5]), len(ids)


def guard_same_reference(vrec, brec, where):
    """Refuse to compare canny records scored against DIFFERENT 2048 condition maps.
    v1.16 replaced the seed-0 c2048 reference with a seed-7 one and rescored results/bench/ but not results/variants/,
    which made an identical decoder look 0.028 better. Records written before the stamp carry no cond_ref: treat an
    absent stamp on one side and a present one on the other as a mismatch, since that is exactly the stale case."""
    if not vrec or not brec:
        return
    vr, br = vrec["adherence"].get("cond_ref"), brec["adherence"].get("cond_ref")
    if vr == br:
        return
    sys.exit(f"ABORT: {where} compares records built on different 2048 condition maps "
             f"(variant cond_ref={vr!r}, baseline cond_ref={br!r}). A record with no stamp predates BENCHMARK v1.16 and "
             f"was scored against the retired seed-0 map. Re-score the variant with the current harness before comparing.")


keymetric = {"canny": ("f1", "higher"), "depth": ("rmse", "lower"), "seg": ("miou", "higher"), "subject": ("dino", "higher")}[a.condition]
rows = [("VAE decode", "vae"), (f"vanilla PiD, K={a.k}", f"pid_k{a.k}")] + ([(f"vanilla PiD teacher, K={a.k}", f"pidt_k{a.k}")] if a.teacher else [])
lines = [f"# {a.name}: {a.controller} / {a.condition} / K={a.k}{' / subset500' if a.subset500 else ''}", "",
         "| row | " + keymetric[0] + " @512 | @2048 | @2048 vs 2048 cond | MUSIQ @512 | MANIQA @512 | LPIPS | FID | pFID |", "|---|---|---|---|---|---|---|---|---|"]


def fmt(r, k, q=False, nd=3):
    if r is None:
        return ""
    v = (r["quality"] if q else r["adherence"]).get(k)
    return "" if v is None else f"{v:.{nd}f}"


def line(label, recs, method):
    r5, r20, rn = pick(recs, method, 512), pick(recs, method, 2048), pick(recs, method, 2048, 2048)
    return f"| {label} | {fmt(r5, keymetric[0])} | {fmt(r20, keymetric[0])} | {fmt(rn, 'f1')} | {fmt(r5, 'musiq', True, 1)} | {fmt(r5, 'maniqa', True)} | {fmt(r5, 'lpips', True)} | {fmt(r5, 'fid', True, 2)} | {fmt(r5, 'pfid', True, 2)} |"


for label, m in rows:
    lines.append(line(label, B, m))
lines.append(line(f"**{a.name}**", V, a.name))
if a.subset500:
    lines.append(""); lines.append("Baseline rows above are the full-set records (n = 5000); the variant is n = 500, so its FID / pFID are NOT comparable with theirs "
                                    "(FID grows with fewer samples). The verdict below uses PAIRED means over the same 500 images.")
lines += ["", f"## Verdict against vanilla PiD at the same K = {a.k} (paired on the same images)", ""]
PB = f"pid_k{a.k}"
v5, v20 = pick(V, a.name, 512), pick(V, a.name, 2048); b5, b20 = pick(B, PB, 512), pick(B, PB, 2048)
if a.condition == "canny":
    guard_same_reference(pick(V, a.name, 2048, 2048), pick(B, PB, 2048, 2048), "the 2048 / c2048 column")
ok = []


def check(name, val, ref, higher=True, tol=0.0, n=None):
    if val is None or ref is None:
        # fail closed: a criterion we could not evaluate is NOT a criterion the variant passed (2026-09-15)
        ok.append(False)
        lines.append(f"- {name}: n/a (no paired data and no record mean) -> FAIL (cannot be evaluated)"); return
    good = (val >= ref - tol) if higher else (val <= ref + tol)
    if tol == 0.0:
        good = (val > ref) if higher else (val < ref)
    ok.append(good); lines.append(f"- {name}: {val:.4f} vs {ref:.4f}{f' (paired, n = {n})' if n else ''} -> {'PASS' if good else 'FAIL'}")


def vals(res, col, cond_res=512, quality=False):
    """paired means when per-image data exist on both sides, else the record means"""
    p = paired(PB, res, cond_res, col=col)
    if p:
        return p
    rv, rb = pick(V, a.name, res, cond_res), pick(B, PB, res, cond_res)
    src = "quality" if quality else "adherence"
    k = RECKEY.get(col) or (col.split("_", 1)[1] if "_" in col else col)
    return (rv and rv[src].get(k), rb and rb[src].get(k), None)


km, direction = keymetric
x = vals(512, KEYCOL); check(f"{km} @512 ({direction} better, strict inequality)", x[0], x[1], direction == "higher", n=x[2])
x = vals(2048, KEYCOL); check(f"{km} @2048 ({direction} better)", x[0], x[1], direction == "higher", n=x[2])
if not a.skip_noref:
    x = vals(512, "musiq", quality=True); check("MUSIQ @512 not lower (tol 0.5)", x[0], x[1], True, tol=0.5, n=x[2])
x = vals(512, "lpips", quality=True); check("LPIPS vs source not worse (tol 0.01)", x[0], x[1], False, tol=0.01, n=x[2])
for res, cond_res in [(512, 512), (2048, 512), (2048, 2048)]:
    if cond_res == 2048 and a.condition != "canny":
        continue
    c = ci(per_image(OUT, a.name, res, cond_res), per_image(BASE, PB, res, cond_res))
    if c:
        lines.append(f"- paired bootstrap, {km} @{res}{' vs 2048 cond' if cond_res == 2048 else ''}: variant minus PiD = {c[0]:+.4f}, 95% CI [{c[1]:+.4f}, {c[2]:+.4f}], n = {c[3]}")
if a.teacher:
    t = paired(f"pidt_k{a.k}", 512); t2 = paired(f"pidt_k{a.k}", 2048)
    if t:
        lines.append(f"- vs TEACHER K={a.k} (paired): {km} @512 {t[0]:.4f} vs {t[1]:.4f}" + (f"; @2048 {t2[0]:.4f} vs {t2[1]:.4f}" if t2 else ""))
lines.append(""); lines.append("**GATE: " + ("PASS" if ok and all(ok) else "FAIL") + "** (all criteria above)" if ok else "**GATE: incomplete**")
lines.append(""); lines.append("Baselines: BASELINES.md (regenerate with bench/make_baselines.py). Protocol: update.md.")
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"\nwrote {OUT / 'REPORT.md'}")
