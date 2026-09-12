# cgd-dev200 — the minimal CGD evaluation loop

A fixed 200-image canny benchmark for iterating on **Conditional Generative Decoding (CGD)**: the same conditioned FLUX latent, decoded by different decoders, scored on how well the decoded pixels honor the condition.

This is the **dev** setting (fast method selection), not the paper benchmark. Numbers here are never reported in the paper; they exist so everyone iterates on the same images, the same latents, and the same scorers. The full protocol lives in the paper repo's `docs/BENCHMARK_v1.md`; this repo implements its `DEV_SETTING.md`.

## Paper benchmark manifests (`bench/`)

`bench/` holds the **frozen manifests of the full paper benchmark** (bench_v1 = BENCHMARK v1.7): `multigen5k` (canny + depth), `ade20k_val2k`, `coco_val5k`, `dreambench750`, the training manifests with a leakage `blocked` column, and the eval-sha blocklist. Images, conditions, and latents are rebuilt with the scripts there (`bench/README.md`); nothing heavy is committed. The dev-200 set below is a subset of `multigen5k` (`dev200_idx` column). **Group-server users: every dataset and scorer is already downloaded; paths, sizes, revisions, and which dataset serves which condition are in `bench/DATA.md`.** **Current state of the benchmark runs, generated-data paths, results, and the remaining-work list: `bench/STATUS.md`.**

## What the comparison is

Every row decodes the **same cached latent** (FLUX.1-dev + OminiControl canny LoRA, 512×512, 28 steps, guidance 3.5, seed 0). Only the decoder changes, so any difference is the decoder's doing:

| Decoder | condition-aware? | generative? | output |
|---|---|---|---|
| VAE decode (FLUX's own) | no | no (deterministic) | 512 |
| vanilla PiD (final latent) | no | yes (pixel diffusion, 4 steps) | 2048 |
| vanilla PiD, early-terminated (latent at step 24/28) | no | yes | 2048 |
| vanilla PiD, early-terminated (latent at step 16/28) | no | yes | 2048 |
| **CGD (your method)** | **yes** | yes | 2048 |

The PiD rows are the baselines a CGD variant must beat. Vanilla PiD on the final latent is the paired ablation ("CGD minus the condition"); the early-terminated rows feed PiD a partially-denoised latent through its sigma-aware adapter: 24/28 (σ≈0.24) is PiD's own recommended operating point, 16/28 is a more aggressive truncation.

## Results (dev-200, OminiControl canny, seed 0)

<!-- RESULTS:BEGIN -->
_Last run: 2026-09-12. Regenerate with `bash scripts/run_all.sh`; this block is written by `scripts/04_fill_readme.py`._

| Decoder | n | Canny F1 @512 (matched, tolerant) ↑ | Canny F1 @2048 (native, tolerant; VAE row via bilinear x4) ↑ | strict F1 @512 (anchor) ↑ | strict F1 @2048 (metric-dominated) | LPIPS ↓ | PSNR ↑ | SSIM ↑ | MUSIQ @512 (matched) ↑ | MUSIQ (native) ↑ | decode s/img ↓ | peak mem GB ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OminiControl + VAE decode (512, native) | 200 | 0.7670 | 0.1550 (bilinear x4) | 0.3647 | 0.0263 | 0.5142 | 11.90 | 0.4394 | 70.40 | 70.40 | 0.033 | 34.5 |
| OminiControl + vanilla PiD (final latent, 2048) | 200 | 0.7781 | 0.7253 | 0.3624 | 0.2217 | 0.5092 | 11.81 | 0.4284 | 71.57 | 61.05 | 0.959 | 23.6 |
| OminiControl + vanilla PiD, early-terminated at 24/28 (2048) | 200 | 0.7133 | 0.6620 | 0.3005 | 0.2033 | 0.5265 | 11.76 | 0.3999 | 72.42 | 61.95 | 0.953 | 23.6 |
| OminiControl + vanilla PiD, early-terminated at 16/28 (2048) | 200 | 0.5378 | 0.5562 | 0.1916 | 0.1622 | 0.6002 | 11.67 | 0.3480 | 74.79 | 66.92 | 0.961 | 23.6 |

Generation (FLUX.1-dev + OminiControl canny LoRA, 28 steps @512, seed 0): 5.11 s/img, shared by every row.
<!-- RESULTS:END -->

How to read it:
- **Canny F1 @512 (matched)** is the number to optimize. Outputs are compared at the VAE's native 512 (PiD/CGD outputs are downsampled from 2048 with `cv2.INTER_AREA`), so a gain cannot come from having more pixels. F1 between `cv2.Canny(gray(output), 100, 200)` and the input condition with a matching tolerance of **one condition pixel** (1 px at 512; BENCHMARK v1.8, the paper metric). The **strict** pixel-exact F1 of the original loop is kept as the 512 anchor column (`canny_f1s_512`); its ranking agrees with the tolerant one at 512.
- **Canny F1 @2048 (native)** is the target: scored at PiD's native output against the condition resampled to 2048 (nearest), with the tolerance scaled to one condition pixel = **4 px at 2048**. The drop from @512 to @2048 within a method is the *resolution gap*: whether the native-resolution detail stayed condition-consistent. **Do not read the strict @2048 column as adherence**: the nearest-resampled reference is 4 px thick while re-extracted edges are 1 to 2 px, so pixel-exact F1 there measures the mismatch (a bicubic x4 upsample of the REAL image scores 0.16 strict); it is reported only for continuity with earlier runs. Canny thresholds are NOT rescaled with resolution, so the native column also requires sharp native-scale edges: the same bicubic x4 real image scores 0.55 tolerant at 2048 (recall loss, edge density 9.5% -> 1.4%), which is intended (a decoder must synthesize the detail, not blur). For the VAE row, which is 512-native, the @2048 value is a **reference** computed on a bilinear x4 upsample of its output (marked "bilinear x4"); the benchmark's never-upsample rule applies to claims, not to this reference.
- **LPIPS ↓ / PSNR ↑ / SSIM ↑** vs the real source image (at 512), computed on the whole RGB image, not on edges. Indicative only: this is *generation from an edge map*, not reconstruction, so colors and textures legitimately differ from the source and PSNR sits around 12 for every method. Use them to catch a decoder that drifts *relative to the others*, not as absolute fidelity.
- **MUSIQ ↑** (no-reference, ~0-100, higher is better): **@512 (matched)** is the comparable number across rows; the native-resolution value is also reported but VAE (512) vs PiD (2048) native scores are not directly comparable.
- **decode s/img, peak mem** on one H200; generation cost is shared by every row.

**The gate for a CGD variant** (from `DEV_SETTING.md`): on these same 200 latents, at the same truncation point K, matched-512 **tolerant** F1 > vanilla PiD's, native-2048 **tolerant** F1 ≥ vanilla PiD's, MUSIQ not lower and LPIPS not worse than vanilla PiD. Beating VAE decode is necessary but not sufficient. (2026-09-12: the gate moved from the strict to the tolerant F1 at both resolutions; the strict @512 column is an anchor, the strict @2048 column is not a gate. If you cloned before this date, `git pull` and re-run `scripts/03_score.py`: the latents, outputs, and id list are unchanged, only the scorer gained the tolerant columns.)

## The fixed set

- Source: HF dataset [`limingcv/MultiGen-20M_canny_eval`](https://huggingface.co/datasets/limingcv/MultiGen-20M_canny_eval), the **5000-image validation split** (the ControlNet++ canny evaluation set; 512×512 LAION-Aesthetics images with captions).
- Selection: `numpy.random.default_rng(0).choice(5000, 200, replace=False)` over the 5 validation shards concatenated in filename order. Frozen in **`dev200/manifest.csv`** (`dev_idx, val_row, sha256 of the image bytes, caption`). Never regenerate it with a different seed under this name; a different set is a different name.
- Condition: `cv2.Canny(grayscale, 100, 200)`, replicated to RGB (the ControlNet annotator defaults; OminiControl's own canny LoRA was trained with the same 100/200 thresholds).
- Prompt: the dataset caption.
- Images and conditions are **not committed** (LAION-derived); `scripts/00_select_dev200.py` rebuilds them bit-exactly from the HF dataset, and the `sha256` column lets you verify you have the same 200. On the group server they are already materialized next to the repo (ask the maintainer for the shared path).

## Environment

Tested inside the group GPU container (4× H200, PyTorch 2.5.1+cu121). Any CUDA environment with `torch`, `diffusers`, `transformers`, `peft`, `pyiqa`, `opencv-python`, `pyarrow`, `pandas` works.

**Layout.** Scripts locate everything relative to one root directory, `$CGD_ROOT` (default: the parent of this repo), overridable per component with `OMINI_ROOT`, `PID_ROOT`, `MULTIGEN_DIR`:
```
$CGD_ROOT/
  cgd-dev200/        this repo
  OminiControl/      clean clone (provides `omini`)
  PiD/               clone + checkpoints/
  data/multigen_canny_eval/   the HF dataset
```

External code (cloned, not vendored):
```bash
cd $CGD_ROOT
git clone --depth 1 https://github.com/Yuanshi9815/OminiControl.git
git clone --depth 1 https://github.com/nv-tlabs/PiD.git
bash cgd-dev200/scripts/env_pid.sh      # PiD's python deps (once)
```
Note: PiD pins `diffusers==0.37.1`, `transformers==4.57.1`, `numpy==1.26.4`; `env_pid.sh` installs them and therefore changes your environment's versions (see "Caveats"). Generation and scoring are verified under these versions.

Weights (downloaded once; not redistributed):
```
hf download black-forest-labs/FLUX.1-dev                                  # generator (gated: accept the license on HF first)
hf download Yuanshi/OminiControl --include "experimental/canny.safetensors" # OminiControl canny LoRA (512)
cd $CGD_ROOT/PiD
hf download nvidia/PiD --local-dir . --include "checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/*"   # 512->2048, 4-step distilled
hf download nvidia/PiD --local-dir . --include "checkpoints/ae.safetensors"                                  # FLUX VAE used by PiD
hf download nvidia/PiD --local-dir . --include "config.json"
```
Run one `--include` pattern per `hf download` call; passing several patterns in one call silently kept only the last one for us.
PiD weights are under the NVIDIA NSCLv1 license (non-commercial research); the PiD **code** is Apache-2.0.

## Run it

```bash
cd $CGD_ROOT/cgd-dev200
bash scripts/run_all.sh            # 00 select -> 01 generate (GPU0) -> 02 PiD decode (GPU1) -> 03 score
```
or step by step:
```bash
python scripts/00_select_dev200.py                                    # ~1 min, CPU: dev200/{manifest.csv,images,canny,captions.json}
CUDA_VISIBLE_DEVICES=0 python scripts/01_generate_latents_omini.py    # ~3-4 s/img: latents/omini_canny/*.pt + outputs/omini_vae/*.png
CUDA_VISIBLE_DEVICES=1 python scripts/02_decode_pid.py                # ~1 s/img/variant: outputs/omini_pid{,_et24,_et16}{,_512}/*.png
CUDA_VISIBLE_DEVICES=1 python scripts/03_score.py                     # results/dev200_summary.{md,csv}, results/dev200_per_image.csv
```
Every step skips outputs that already exist; add `--overwrite` to redo, `--limit N` to smoke-test on N images.

### What is cached, and why it matters
`latents/omini_canny/{idx}.pt` holds, per image: the **final clean latent** `x_0` (1×16×64×64, fp16, in diffusers' scaled latent space, i.e. exactly what PiD's `extract_latent` produces), its `sigma` (0.0), the **early-terminated latents** `xt16` / `xt24` after 16 and 24 of 28 steps with their `sigma16` / `sigma24`, and the caption/seed/steps. Generation (28 FLUX steps) is the slow part; decoding is fast. **Every decoder variant must read these files and never regenerate latents**, otherwise the comparison is no longer paired.

## Adding your CGD variant

1. Write `scripts/02_decode_<name>.py` that loads `latents/omini_canny/*.pt`, decodes each `latent` (and optionally `xt24`) to 2048, and writes `outputs/<name>/{idx}.png` plus the INTER_AREA 512 view to `outputs/<name>_512/{idx}.png`. Copy `02_decode_pid.py`; the condition image for CGD is `dev200/canny/{idx}.png`.
2. `python scripts/03_score.py --variants omini_vae,omini_pid,omini_pid_et24,<name>`.
3. Log one line per run somewhere shared: `variant, F1@512, F1@2048, LPIPS, PSNR, MUSIQ, s/img, notes`.
4. Promote only if it passes the gate above. Then L2: add depth (the other released OminiControl spatial LoRA) and the other controllers per `DEV_SETTING.md`.

## Caveats (read once)

- **Dependency pins:** `env_pid.sh` installs PiD's pinned `diffusers 0.37.1 / transformers 4.57.1 / numpy 1.26.4`. Generation and scoring were verified under these versions. If another project in the same environment needs newer versions, give PiD its own venv.
- **Latents are not bit-reproducible across runs** (GPU nondeterminism, and sensitivity to the diffusers/transformers versions): regenerating shifts a latent slightly (in our check, max |Δ| ≈ 0.4 on a std-0.69 latent). This is exactly why the cached latents are shared and every decoder must read them rather than regenerate; all rows in the results table come from a single generation run.
- **Strict F1** has no pixel tolerance, so absolute values look low; only relative comparisons between rows matter.
- **Same seed (0) for every image**: the initial noise is identical across images; this is deliberate for reproducibility and is fine for a paired decoder comparison.
- **Early-terminated PiD** rows decode a *different* latent (σ≈0.24 at 24/28, larger at 16/28) than the VAE row, so they are reference points for PiD's early-termination mode, not part of the paired swap. The paired swap is VAE vs PiD(final) vs CGD(final).
- The 4-step distilled PiD ignores `shift`/`cfg` (it uses its student timestep list); `--ckpt-type 2kto4k_v1pt5` switches to the multi-resolution v1.5 checkpoint if needed.

## Layout
```
dev200/            manifest.csv, captions.json (committed); images/, canny/ (rebuilt by 00, not committed)
scripts/           00_select_dev200.py  01_generate_latents_omini.py  02_decode_pid.py  03_score.py  run_all.sh  env_pid.sh
latents/           cached latents (not committed; shared on the group server)
outputs/           decoded images per variant (not committed)
results/           dev200_summary.md / .csv (committed), per-image csv + run logs (not committed)
```
