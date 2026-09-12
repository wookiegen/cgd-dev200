"""Frozen scorers of the CGD benchmark (BENCHMARK_v1.md Section 5; SCORING_HARNESS.md). One class per condition, lazy model loading,
identical code path for every method. All image inputs are uint8 RGB numpy arrays at the SCORING resolution (512 view or 2048 native).

  CannyF1      cv2.Canny(gray, 100, 200) on the output vs the input edge map (nearest-resampled to the scoring resolution); F1 with a
               one-condition-pixel tolerance (v1.8, the paper metric) plus the strict pixel F1 (anchor to published 512 numbers).
  DepthScorer  Intel/dpt-large on the output; reference = the input DPT depth (float, 512) resized to the scoring resolution and min-max
               normalized to [0, 255]; the prediction is least-squares scale+shift aligned to the reference; MSE and RMSE in those units.
  SegScorer    facebook/mask2former-swin-large-ade-semantic on the output -> ids 0..149 (+1 = ADE20K labels 1..150); dataset-level mIoU
               over classes with non-empty union, GT label 0 ignored; reported x100.
  LayoutScorer IDEA-Research/grounding-dino-base with class-name prompts (MIGC / COCO-MIG protocol): per GT instance the max-IoU detection of
               the same class; success at IoU >= 0.5; Instance Success Rate and mean IoU; box_threshold 0.35, text_threshold 0.25.
  NoRef        pyiqa musiq-paq2piq, niqe, maniqa, qalign (per image, mean). DeQA / UniPercept / VisualQuality-R1 live in PiD's evaluation
               module and are added by harness.py when --vlm is given.
  Recon        pyiqa psnr, ssim, lpips, dists vs the real 512 source (indicative; only where a paired real image exists).
  SubjectScorer DreamBench protocol: DINO (facebook/dino-vits16 CLS cosine vs every real image of the subject, mean), CLIP-I
               (openai/clip-vit-large-patch14 image cosine, mean), CLIP-T (CLIP image vs prompt text cosine); whole images at the 512 view.
  fid / pfid   clean-fid, mode "clean"; pFID = one random 299x299 crop per image with a fixed per-sample seed, at the 512 view.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_tensor01(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].float().div(255).to(DEVICE)


def resize(arr: np.ndarray, size: int, interp) -> np.ndarray:
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    return cv2.resize(arr, (size, size), interpolation=interp)


# ----------------------------------------------------------------------------- canny
class CannyF1:
    name = "canny"

    @staticmethod
    def edges(rgb: np.ndarray) -> np.ndarray:
        return cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 100, 200) > 0

    @staticmethod
    def f1_strict(pred: np.ndarray, ref: np.ndarray) -> float:
        tp = np.logical_and(pred, ref).sum(); fp = np.logical_and(pred, ~ref).sum(); fn = np.logical_and(~pred, ref).sum()
        return float(2 * tp / max(2 * tp + fp + fn, 1))

    @staticmethod
    def f1_tolerant(pred: np.ndarray, ref: np.ndarray, tol: int) -> float:
        """Edge F1 with a matching tolerance of `tol` pixels (BSDS-style): a predicted edge pixel is correct if a reference edge lies within
        tol pixels, and a reference pixel is recalled if a predicted edge lies within tol pixels."""
        if tol <= 0:
            return CannyF1.f1_strict(pred, ref)
        k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
        ref_d = cv2.dilate(ref.astype(np.uint8), k) > 0; pred_d = cv2.dilate(pred.astype(np.uint8), k) > 0
        prec = np.logical_and(pred, ref_d).sum() / max(pred.sum(), 1); rec = np.logical_and(ref, pred_d).sum() / max(ref.sum(), 1)
        return float(2 * prec * rec / max(prec + rec, 1e-9))

    def score(self, out_rgb: np.ndarray, cond_rgb512: np.ndarray) -> dict:
        """BENCHMARK v1.8: `f1` = tolerant F1 with a tolerance of ONE CONDITION PIXEL (1 px at 512, 4 px at 2048), the paper's edge metric;
        `f1_strict` = pixel-exact F1 (the ControlNet++ convention at 512; kept as the anchor to published numbers). The strict score at
        2048 mostly measures the 4-px-thick nearest-upsampled reference against thin re-extracted edges (a bicubic upsample of the REAL
        image scores 0.15 strict), which is why it is not the paper metric at native resolution."""
        res = out_rgb.shape[0]; cres = cond_rgb512.shape[0]
        e = (cond_rgb512[..., 0] > 0).astype(np.uint8)
        if cres <= res:        # condition coarser than (or equal to) the output: nearest resample, one condition pixel = res // cres output pixels
            ref = resize(e, res, cv2.INTER_NEAREST) > 0
        else:                  # condition finer than the output (native-condition track scored at the 512 view): a block with any edge is an edge
            ref = cv2.resize(e.astype(np.float32), (res, res), interpolation=cv2.INTER_AREA) > 0
        pred = self.edges(out_rgb)
        tol = max(1, res // cres)
        return {"f1": self.f1_tolerant(pred, ref, tol), "f1_strict": self.f1_strict(pred, ref), "tol_px": tol}


# ----------------------------------------------------------------------------- depth
class DepthScorer:
    name = "depth"
    MODEL = "Intel/dpt-large"

    def __init__(self):
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        self.proc = DPTImageProcessor.from_pretrained(self.MODEL)
        self.model = DPTForDepthEstimation.from_pretrained(self.MODEL).to(DEVICE).eval()

    @torch.no_grad()
    def predict(self, rgb: np.ndarray, size: int) -> np.ndarray:
        from PIL import Image
        inp = self.proc(images=Image.fromarray(rgb), return_tensors="pt").to(DEVICE)
        pred = self.model(**inp).predicted_depth
        return torch.nn.functional.interpolate(pred.unsqueeze(1), size=(size, size), mode="bicubic", align_corners=False)[0, 0].float().cpu().numpy()

    @staticmethod
    def normalize255(d: np.ndarray) -> np.ndarray:
        lo, hi = float(d.min()), float(d.max())
        return np.zeros_like(d) if hi - lo < 1e-8 else (d - lo) / (hi - lo) * 255.0

    def score(self, out_rgb: np.ndarray, ref_depth_raw512: np.ndarray) -> dict:
        res = out_rgb.shape[0]
        ref = self.normalize255(resize(ref_depth_raw512.astype(np.float32), res, cv2.INTER_CUBIC))
        pred = self.predict(out_rgb, res)
        # least-squares scale + shift of pred onto ref
        A = np.stack([pred.ravel(), np.ones(pred.size)], 1)
        sol, *_ = np.linalg.lstsq(A, ref.ravel(), rcond=None)
        aligned = A @ sol
        err = aligned - ref.ravel()
        mse = float(np.mean(err ** 2))
        return {"mse": mse, "rmse": float(np.sqrt(mse))}


# ----------------------------------------------------------------------------- segmentation
class SegScorer:
    name = "seg"
    MODEL = "facebook/mask2former-swin-large-ade-semantic"
    N = 150

    def __init__(self):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.proc = AutoImageProcessor.from_pretrained(self.MODEL)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.MODEL).to(DEVICE).eval()
        self.inter = np.zeros(self.N + 1, dtype=np.int64); self.union = np.zeros(self.N + 1, dtype=np.int64)

    @torch.no_grad()
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        inp = self.proc(images=Image.fromarray(rgb), return_tensors="pt").to(DEVICE)
        out = self.model(**inp)
        seg = self.proc.post_process_semantic_segmentation(out, target_sizes=[rgb.shape[:2]])[0]
        return seg.cpu().numpy().astype(np.int64) + 1          # 1..150, same indexing as the GT label maps

    def score(self, out_rgb: np.ndarray, gt_label512: np.ndarray) -> dict:
        """Accumulates dataset-level intersections/unions; returns the per-image mIoU (over classes present in GT or pred) for the CSV."""
        res = out_rgb.shape[0]
        gt = resize(gt_label512.astype(np.uint8), res, cv2.INTER_NEAREST).astype(np.int64)
        pred = self.predict(out_rgb)
        valid = gt > 0
        g, p = gt[valid], pred[valid]
        inter = np.bincount(g[g == p], minlength=self.N + 1)
        area_g = np.bincount(g, minlength=self.N + 1); area_p = np.bincount(p, minlength=self.N + 1)
        union = area_g + area_p - inter
        self.inter += inter; self.union += union
        m = union[1:] > 0
        return {"miou_img": float(100 * np.mean(inter[1:][m] / union[1:][m])) if m.any() else float("nan")}

    def dataset_miou(self) -> float:
        m = self.union[1:] > 0
        return float(100 * np.mean(self.inter[1:][m] / self.union[1:][m]))

    def reset(self):
        self.inter[:] = 0; self.union[:] = 0


# ----------------------------------------------------------------------------- layout (GroundingDINO, MIGC protocol)
def box_iou_xywh(a, b) -> float:
    ax0, ay0, aw, ah = a; bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0); ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    return inter / max(aw * ah + bw * bh - inter, 1e-9)


class LayoutScorer:
    name = "bbox"
    MODEL = "IDEA-Research/grounding-dino-base"
    BOX_T, TEXT_T = 0.35, 0.25

    def __init__(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(self.MODEL)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.MODEL).to(DEVICE).eval()

    @torch.no_grad()
    def detect(self, rgb: np.ndarray, class_names: list[str]) -> list[tuple[str, float, tuple]]:
        """Returns [(phrase, score, (x, y, w, h))] at the image resolution."""
        from PIL import Image
        prompt = ". ".join(class_names) + "."
        inp = self.proc(images=Image.fromarray(rgb), text=prompt, return_tensors="pt").to(DEVICE)
        out = self.model(**inp)
        try:
            res = self.proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=self.BOX_T, text_threshold=self.TEXT_T,
                                                                   target_sizes=[rgb.shape[:2]])[0]
        except TypeError:   # older signature
            res = self.proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=self.BOX_T, text_threshold=self.TEXT_T,
                                                                   target_sizes=[rgb.shape[:2]])[0]
        labels = res.get("text_labels", res.get("labels"))
        dets = []
        for lab, sc, bx in zip(labels, res["scores"].tolist(), res["boxes"].tolist()):
            x0, y0, x1, y1 = bx
            dets.append((str(lab), float(sc), (x0, y0, x1 - x0, y1 - y0)))
        return dets

    @staticmethod
    def phrase_matches(phrase: str, name: str) -> bool:
        p, n = phrase.lower().strip(), name.lower()
        return p == n or n in p or p in n

    def score(self, out_rgb: np.ndarray, gt_boxes: list[dict]) -> dict:
        """gt_boxes: [{'name': str, 'bbox_xywh_crop': [x, y, w, h]}] in 512 crop coords. Returns per-instance results aggregated per image."""
        res = out_rgb.shape[0]; s = res / 512.0
        names = sorted({b["name"] for b in gt_boxes})
        dets = self.detect(out_rgb, names) if names else []
        ious = []
        for b in gt_boxes:
            g = tuple(v * s for v in b["bbox_xywh_crop"])
            cand = [box_iou_xywh(g, d[2]) for d in dets if self.phrase_matches(d[0], b["name"])]
            ious.append(max(cand) if cand else 0.0)
        ious = np.array(ious, dtype=np.float64)
        return {"n_inst": int(len(ious)), "n_success": int((ious >= 0.5).sum()), "sum_iou": float(ious.sum()),
                "sr_img": float((ious >= 0.5).mean()) if len(ious) else float("nan"), "miou_img": float(ious.mean()) if len(ious) else float("nan")}


# ----------------------------------------------------------------------------- quality
class NoRef:
    NAMES = {"musiq": "musiq-paq2piq", "niqe": "niqe", "maniqa": "maniqa", "qalign": "qalign"}

    @staticmethod
    def _patch_qalign_loader():
        """transformers (torch < 2.6) refuses the .bin shards of q-future/one-align and, given the repo id, ignores locally converted
        safetensors because it checks the Hub first. Loading from the local snapshot directory makes it take the safetensors shards
        (convert once with tools in the paper repo / DATA.md)."""
        try:
            import pyiqa.archs.qalign_arch as QA
            from huggingface_hub import snapshot_download
            local = snapshot_download("q-future/one-align", local_files_only=True)
            if not os.path.exists(os.path.join(local, "model.safetensors.index.json")):
                return
            orig = QA.MPLUGOwl2LlamaForCausalLM.from_pretrained

            def patched(name, *args, **kw):
                return orig(local if name == "q-future/one-align" else name, *args, **kw)
            QA.MPLUGOwl2LlamaForCausalLM.from_pretrained = staticmethod(patched)
        except Exception as e:  # noqa: BLE001
            print(f"[NoRef] qalign loader patch skipped: {type(e).__name__}: {str(e)[:80]}", flush=True)

    def __init__(self, keys=("musiq", "niqe", "maniqa", "qalign")):
        import pyiqa
        if "qalign" in keys:
            self._patch_qalign_loader()
        self.m = {}
        for k in keys:
            try:
                self.m[k] = pyiqa.create_metric(self.NAMES[k], device=DEVICE)
            except Exception as e:  # noqa: BLE001  (e.g. Q-Align weights not loadable in this environment): score the rest, say so
                print(f"[NoRef] {k} unavailable, skipped: {type(e).__name__}: {str(e)[:100]}", flush=True)

    @torch.no_grad()
    def score(self, rgb: np.ndarray) -> dict:
        t = to_tensor01(rgb)
        return {k: float(m(t)) for k, m in self.m.items()}


class Recon:
    def __init__(self):
        import pyiqa
        self.m = {k: pyiqa.create_metric(k, device=DEVICE) for k in ["psnr", "ssim", "lpips", "dists"]}

    @torch.no_grad()
    def score(self, rgb512: np.ndarray, ref512: np.ndarray) -> dict:
        a, b = to_tensor01(rgb512), to_tensor01(ref512)
        return {k: float(m(a, b)) for k, m in self.m.items()}


class SubjectScorer:
    """DreamBench protocol (DreamBooth; the UNO / OminiControl re-reports): DINO = cosine between the ViT-S/16 CLS embedding of the output and
    of each real image of the subject, averaged; CLIP-I = the same with CLIP ViT-L/14 image embeddings; CLIP-T = cosine between the CLIP
    image embedding of the output and the CLIP text embedding of the prompt (class-name prompt, no unique token). Whole images, no cropping.
    Real images = every image of the subject in the DreamBooth dataset (5 or 6 per subject); embeddings cached per subject."""
    DINO = "facebook/dino-vits16"; CLIP = "openai/clip-vit-large-patch14"

    def __init__(self, dataset_root: Path | None = None):
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
        self.root = Path(dataset_root or os.environ.get("CGD_RAW_ROOT", "/data/wookiekim/cgd/data")) / "dreambench" / "dataset"
        self.dino = AutoModel.from_pretrained(self.DINO).to(DEVICE).eval(); self.dino_proc = AutoImageProcessor.from_pretrained(self.DINO)
        self.clip = CLIPModel.from_pretrained(self.CLIP).to(DEVICE).eval(); self.clip_proc = CLIPProcessor.from_pretrained(self.CLIP)
        self._real = {}; self._text = {}

    @torch.no_grad()
    def embed(self, imgs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image
        pil = [Image.fromarray(x) for x in imgs]
        d = self.dino(**self.dino_proc(images=pil, return_tensors="pt").to(DEVICE)).last_hidden_state[:, 0]
        c = self.clip.get_image_features(**self.clip_proc(images=pil, return_tensors="pt").to(DEVICE))
        return torch.nn.functional.normalize(d.float(), dim=-1), torch.nn.functional.normalize(c.float(), dim=-1)

    def real(self, subject: str):
        if subject not in self._real:
            from PIL import Image
            files = sorted(p for p in (self.root / subject).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            self._real[subject] = self.embed([np.array(Image.open(p).convert("RGB")) for p in files])
        return self._real[subject]

    @torch.no_grad()
    def text(self, prompt: str) -> torch.Tensor:
        if prompt not in self._text:
            t = self.clip.get_text_features(**self.clip_proc(text=[prompt], return_tensors="pt", padding=True).to(DEVICE))
            self._text[prompt] = torch.nn.functional.normalize(t.float(), dim=-1)
        return self._text[prompt]

    def score(self, out_rgb: np.ndarray, subject: str, prompt: str) -> dict:
        d, c = self.embed([out_rgb]); rd, rc = self.real(subject)
        return {"dino": float((d @ rd.T).mean()), "clip_i": float((c @ rc.T).mean()), "clip_t": float((c @ self.text(prompt).T).mean())}


def crop299(rgb: np.ndarray, sample_id: str, seed: int = 0) -> np.ndarray:
    """One deterministic random 299x299 crop per sample (seed derived from sample_id) for pFID."""
    h, w = rgb.shape[:2]
    rng = np.random.default_rng(int(hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()[:8], 16))
    y = int(rng.integers(0, h - 299 + 1)); x = int(rng.integers(0, w - 299 + 1))
    return rgb[y:y + 299, x:x + 299]


def compute_fid(dir_a: str, dir_b: str) -> float:
    from cleanfid import fid
    return float(fid.compute_fid(dir_a, dir_b, mode="clean", device=torch.device(DEVICE), num_workers=8, verbose=False))
