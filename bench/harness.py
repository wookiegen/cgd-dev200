"""CGD scoring harness (SCORING_HARNESS.md). Scores one method's images on one eval set at one resolution with the frozen scorers.

  python harness.py --method <label> --split <multigen5k|ade20k_val2k|coco_val5k|dreambench750> --gen-dir <dir of <id>.png> --res <512|2048>
                    [--condition canny,depth | seg | bbox | subject]   (default: all conditions of the split; dreambench750 has no paired real image, so recon / fid are skipped there)
                    [--gen-dir-512 <dir>]  precomputed INTER_AREA 512 view of a 2048 gen-dir (else computed on the fly)
                    [--metrics adherence,noref,recon,fid,vlm]   (default: adherence,noref,recon,fid; recon/fid only at res 512; vlm = DeQA, VQ-R1, UniPercept, slow)
                    [--controller <label>] [--limit N] [--out-dir results/bench]
The gen-dir is any directory of <sample_id>.png at the method's native resolution; a 512-native method scored at --res 2048 is refused
(never upsample). Conditions, GT, and real images come from $CGD_BENCH_ROOT/<split>/. The Real row = --gen-dir <split>/images --method real.
Writes <out-dir>/<split>/<method>@<res>.json (one record per condition, spec schema) and <method>@<res>_per_image.csv.
"""
import argparse
import csv
import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common import OUT_ROOT, REPO_BENCH
import scorers as S

CONDS = {"multigen5k": ["canny", "depth"], "ade20k_val2k": ["seg"], "coco_val5k": ["bbox"], "dreambench750": ["subject"], "lsdir1k": ["canny"]}
NO_PAIRED_REAL = {"dreambench750"}   # no paired source image: recon / fid are not defined there

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--split", required=True, choices=list(CONDS))
ap.add_argument("--gen-dir", required=True); ap.add_argument("--gen-dir-512", default=None)
ap.add_argument("--res", type=int, required=True, choices=[512, 2048])
ap.add_argument("--condition", default=None); ap.add_argument("--metrics", default="adherence,noref,recon,fid")
ap.add_argument("--controller", default=""); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out-dir", default=str(REPO_BENCH.parent / "results" / "bench"))
ap.add_argument("--vlm-batch", type=int, default=16, help="images per forward for the VLM scorers (metrics keyword 'vlm': DeQA, VisualQuality-R1, UniPercept IAA/IQA via PiD's evaluation module)")
ap.add_argument("--subset500", action="store_true", help="score only the manifest rows with subset500 == 1 (file tag gets '.subset500'; FID against the same subset of real images)")
ap.add_argument("--cond-res", type=int, default=512, help="canny only: resolution of the condition used as the scoring reference; 2048 = the native-condition track (conditions/canny2048, tolerance one condition pixel = 1 px at 2048; file tag '.cond2048')")
a = ap.parse_args()

conds = a.condition.split(",") if a.condition else CONDS[a.split]
metrics = set(a.metrics.split(","))
if a.res == 2048 or a.split in NO_PAIRED_REAL:
    metrics -= {"recon", "fid"}
rows = list(csv.DictReader(open(REPO_BENCH / a.split / "manifest.csv")))
if a.subset500:
    rows = [r for r in rows if r.get("subset500") == "1"]
rows = rows[: a.limit or None]
gen = Path(a.gen_dir); gen512 = Path(a.gen_dir_512) if a.gen_dir_512 else None
bench = OUT_ROOT / a.split
out_dir = Path(a.out_dir) / a.split; out_dir.mkdir(parents=True, exist_ok=True)
DEFAULT_METRICS = {"adherence", "noref", "recon", "fid"}
tag = f"{a.method}@{a.res}"
if a.controller:                      # results of different controllers must not share a file name
    tag = f"{a.controller}.{tag}"
file_tag = tag if metrics == (DEFAULT_METRICS - ({"recon", "fid"} if (a.res == 2048 or a.split in NO_PAIRED_REAL) else set())) else f"{tag}.{'-'.join(sorted(metrics))}"
if conds != CONDS[a.split]:          # a run restricted to a subset of the split's conditions gets its own file (two condition runs must not overwrite each other)
    file_tag += "." + "-".join(conds)
if a.subset500:
    file_tag += ".subset500"
if a.cond_res != 512:
    file_tag += f".cond{a.cond_res}"
CANNY_DIR = "canny" if a.cond_res == 512 else f"canny{a.cond_res}"

# probe native resolution
probe = np.array(Image.open(gen / f"{rows[0]['sample_id']}.png"))
native = probe.shape[0]
if a.res > native:
    raise SystemExit(f"refusing to score a {native}-native method at {a.res} (never upsample); use --res {native}")
print(f"{tag} on {a.split}: {len(rows)} rows, native {native}, conds {conds}, metrics {sorted(metrics)}", flush=True)


def load_view(sid: str) -> np.ndarray:
    if a.res == native:
        return np.array(Image.open(gen / f"{sid}.png").convert("RGB"))
    if gen512 is not None and (gen512 / f"{sid}.png").exists():
        return np.array(Image.open(gen512 / f"{sid}.png").convert("RGB"))
    nat = np.array(Image.open(gen / f"{sid}.png").convert("RGB"))
    return cv2.resize(nat, (a.res, a.res), interpolation=cv2.INTER_AREA)


# scorers
sc = {}
if "adherence" in metrics:
    if "canny" in conds: sc["canny"] = S.CannyF1()
    if "depth" in conds: sc["depth"] = S.DepthScorer()
    if "seg" in conds: sc["seg"] = S.SegScorer()
    if "bbox" in conds: sc["bbox"] = S.LayoutScorer()
    if "subject" in conds: sc["subject"] = S.SubjectScorer()
noref = S.NoRef() if "noref" in metrics else None
recon = S.Recon() if "recon" in metrics else None
boxes = {}
if "bbox" in conds:
    for l in open(REPO_BENCH / a.split / "boxes.jsonl"):
        d = json.loads(l); boxes[d["sample_id"]] = d["boxes"]
vlm = None
if "vlm" in metrics:
    import sys
    pid_root = os.environ.get("PID_ROOT", "/data/wookiekim/cgd/PiD"); sys.path.insert(0, pid_root); cwd = os.getcwd(); os.chdir(pid_root)
    from pid._src.evaluations.metrics import image_metrics as IM  # noqa: E402
    os.chdir(cwd)
    import transformers
    try:
        import flash_attn  # noqa: F401
    except ImportError:   # PiD's VisualQuality-R1 loader hardcodes flash_attention_2; fall back to SDPA when flash_attn is absent
        _orig_fp = transformers.AutoModelForVision2Seq.from_pretrained.__func__

        def _fp(cls, *args, **kw):
            if kw.get("attn_implementation") == "flash_attention_2":
                kw["attn_implementation"] = "sdpa"
            return _orig_fp(cls, *args, **kw)
        transformers.AutoModelForVision2Seq.from_pretrained = classmethod(_fp)
    vlm = {"deqa": IM.DeQAScore(device=S.DEVICE), "vqr1": IM.VisualQualityR1(device=S.DEVICE),
           "unipercept_iaa": IM.UniPerceptIAA(device=S.DEVICE), "unipercept_iqa": IM.UniPerceptIQA(device=S.DEVICE)}
    vlm_buf = []   # (rec, view) pairs scored in chunks of --vlm-batch


def flush_vlm():
    if not vlm_buf:
        return
    views = [v for _, v in vlm_buf]
    for name, m in vlm.items():
        try:
            vals = m.compute_batch_list(views)          # numpy HWC uint8 inputs
        except Exception as e:  # noqa: BLE001
            vals = [float("nan")] * len(views); vlm_buf[0][0].setdefault("_vlm_err", f"{name}: {str(e)[:80]}")
        for (rec_, _), v in zip(vlm_buf, vals):
            rec_[name] = float(v)
    vlm_buf.clear()

per_image = []
crop_dir_gen = crop_dir_real = None
if "fid" in metrics:
    tmp = Path(tempfile.mkdtemp(prefix="pfid_"))
    crop_dir_gen, crop_dir_real = tmp / "gen", tmp / "real"; crop_dir_gen.mkdir(); crop_dir_real.mkdir()
t0 = time.time()
for k, r in enumerate(rows):
    sid = r["sample_id"]
    view = load_view(sid)
    rec = {"sample_id": sid}
    if "canny" in sc:
        cond = np.array(Image.open(bench / "conditions" / CANNY_DIR / f"{sid}.png").convert("RGB"))
        rec.update({f"canny_{kk}": v for kk, v in sc["canny"].score(view, cond).items()})
    if "depth" in sc:
        ref = np.load(bench / "conditions" / "depth_raw" / f"{sid}.npy").astype(np.float32)
        rec.update({f"depth_{kk}": v for kk, v in sc["depth"].score(view, ref).items()})
    if "seg" in sc:
        gt = np.array(Image.open(bench / "labels" / f"{sid}.png"))
        rec.update({f"seg_{kk}": v for kk, v in sc["seg"].score(view, gt).items()})
    if "bbox" in sc:
        gb = boxes.get(sid, [])
        if gb:
            rec.update({f"bbox_{kk}": v for kk, v in sc["bbox"].score(view, gb).items()})
    if "subject" in sc:
        rec.update({f"subject_{kk}": v for kk, v in sc["subject"].score(view, r["subject"], r["prompt"]).items()})
    if noref is not None:
        rec.update(noref.score(view))
    if vlm is not None:
        vlm_buf.append((rec, view))
        if len(vlm_buf) >= a.vlm_batch:
            flush_vlm()
    if recon is not None:
        real = np.array(Image.open(bench / "images" / f"{sid}.png").convert("RGB"))
        rec.update(recon.score(view, real))
    if crop_dir_gen is not None:
        real = np.array(Image.open(bench / "images" / f"{sid}.png").convert("RGB"))
        Image.fromarray(S.crop299(view, sid)).save(crop_dir_gen / f"{sid}.png")
        Image.fromarray(S.crop299(real, sid)).save(crop_dir_real / f"{sid}.png")
    per_image.append(rec)
    if (k + 1) % 250 == 0:
        print(f"  [{k+1}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)
if vlm is not None:
    flush_vlm()

# aggregate
def mean(key):
    v = [x[key] for x in per_image if key in x and x[key] == x[key]]
    return float(np.mean(v)) if v else None

quality = {}
for key in ["musiq", "niqe", "maniqa", "qalign", "deqa", "vqr1", "unipercept_iaa", "unipercept_iqa", "psnr", "ssim", "lpips", "dists"]:
    m = mean(key)
    if m is not None:
        quality[key] = m
if crop_dir_gen is not None:
    view_dir = gen if a.res == native else None
    if view_dir is None:  # materialize the 512 view for FID
        view_dir = Path(tempfile.mkdtemp(prefix="view512_"))
        for r in rows:
            Image.fromarray(load_view(r["sample_id"])).save(view_dir / f"{r['sample_id']}.png")
    real_dir = bench / "images"
    if a.method != "real":
        if a.limit or a.subset500:   # FID over a subset: compare against the same subset of real images
            sub = Path(tempfile.mkdtemp(prefix="realsub_"))
            for r in rows:
                shutil.copy(real_dir / f"{r['sample_id']}.png", sub / f"{r['sample_id']}.png")
            real_dir = sub
        quality["fid"] = S.compute_fid(str(view_dir), str(real_dir))
        quality["pfid"] = S.compute_fid(str(crop_dir_gen), str(crop_dir_real))
    shutil.rmtree(crop_dir_gen.parent, ignore_errors=True)

records = []
for c in conds:
    adh = {}
    if c == "canny" and "canny" in sc: adh = {"f1": mean("canny_f1"), "f1_strict": mean("canny_f1_strict"), "tol_px": int(max(1, a.res // a.cond_res)), "cond_res": a.cond_res,
                                              "precision": mean("canny_precision"), "recall": mean("canny_recall"), "edge_density": mean("canny_edge_density")}
    if c == "depth" and "depth" in sc: adh = {"mse": mean("depth_mse"), "rmse": mean("depth_rmse")}
    if c == "seg" and "seg" in sc: adh = {"miou": sc["seg"].dataset_miou(), "miou_img_mean": mean("seg_miou_img")}
    if c == "bbox" and "bbox" in sc:
        n_inst = sum(x.get("bbox_n_inst", 0) for x in per_image); n_succ = sum(x.get("bbox_n_success", 0) for x in per_image)
        adh = {"sr": 100.0 * n_succ / max(n_inst, 1), "miou": 100.0 * sum(x.get("bbox_sum_iou", 0.0) for x in per_image) / max(n_inst, 1),
               "n_instances": n_inst, "n_images_with_boxes": sum(1 for x in per_image if x.get("bbox_n_inst", 0) > 0)}
    if c == "subject" and "subject" in sc: adh = {"dino": mean("subject_dino"), "clip_i": mean("subject_clip_i"), "clip_t": mean("subject_clip_t")}
    records.append({"method": a.method, "controller": a.controller, "condition": c, "split": a.split, "res": a.res,
                    "native_res": native, "subset": "subset500" if a.subset500 else "", "adherence": adh, "quality": quality, "n": len(per_image),
                    "gen_dir": str(gen), "time_s": round(time.time() - t0, 1)})
json.dump(records, open(out_dir / f"{file_tag}.json", "w"), indent=1)
keys = sorted({k for x in per_image for k in x}, key=lambda k: (k != "sample_id", k))
with open(out_dir / f"{file_tag}_per_image.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(per_image)
for rec in records:
    print(json.dumps({k: rec[k] for k in ["method", "condition", "res", "adherence", "quality", "n"]}))
print(f"wrote {out_dir / (file_tag + '.json')}")
