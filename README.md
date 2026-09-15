# cgd-dev200 — the minimal CGD evaluation loop

A fixed 200-image canny benchmark for iterating on **Conditional Generative Decoding (CGD)**: the same conditioned FLUX latent, decoded by different decoders, scored on how well the decoded pixels honor the condition.

**Start here (2026-09-12):** `BASELINES.md` = the numbers a variant has to beat (auto-generated from `results/bench/`); `bench/eval_variant.py` = one command that scores your 2048 outputs under the paper protocol and prints PASS / FAIL against those baselines; `update.md` = the protocol and its rationale; `BRIEF_ko.md` = the same in Korean, short.

This is the **dev** setting (fast method selection), not the paper benchmark. Numbers here are never reported in the paper; they exist so everyone iterates on the same images, the same latents, and the same scorers. The full protocol lives in the paper repo's `docs/BENCHMARK_v1.md`; this repo implements its `DEV_SETTING.md`.

## Paper benchmark manifests (`bench/`)

`bench/` holds the **frozen manifests of the full paper benchmark** (bench_v1 = BENCHMARK v1.7): `multigen5k` (canny + depth), `ade20k_val2k`, `coco_val5k`, `dreambench750`, the training manifests with a leakage `blocked` column, and the eval-sha blocklist. Images, conditions, and latents are rebuilt with the scripts there (`bench/README.md`); nothing heavy is committed. The dev-200 set below is a subset of `multigen5k` (`dev200_idx` column). **Group-server users: every dataset and scorer is already downloaded; paths, sizes, revisions, and which dataset serves which condition are in `bench/DATA.md`.** **Current state of the benchmark runs, generated-data paths, results, and the remaining-work list: `bench/STATUS.md`.**

## What the comparison is

Every row decodes the **same cached latent** (FLUX.1-dev + OminiControl canny LoRA, 512×512, 28 steps, guidance 3.5, seed 0). Only the decoder changes, so any difference is the decoder's doing:

| Decoder | condition-aware? | generative? | output | route to 2048 |
|---|---|---|---|---|
| VAE decode (FLUX's own) | no | no (deterministic) | 512 | interpolation (bicubic x4, ‡) |
| OminiControl generating at 2048 + VAE decode | no | no | 2048 | native route (FLUX at 4 MP; slow) |
| vanilla PiD student (final latent; K=24; K=16) | no | yes (pixel diffusion, 4 steps) | 2048 | pixel decoder from the 512 latent |
| vanilla PiD teacher (final latent; K=24; K=16) | no | yes (undistilled, 25 steps, CFG 5) | 2048 | pixel decoder from the 512 latent |
| **CGD (your method)** | **yes** | yes | 2048 | pixel decoder from the 512 latent (+ the condition, at 512 or at 2048) |

The PiD rows are the baselines a CGD variant must beat. Vanilla PiD on the final latent is the paired ablation ("CGD minus the condition"); the early-terminated rows feed PiD a partially-denoised latent through its sigma-aware adapter: 24/28 (σ≈0.24) is PiD's own recommended operating point, 16/28 is a more aggressive truncation.

Queued after the current GPU work (BENCHMARK v1.13): a validation of the c2048 column on 1000 REAL 2048 photographs (DIV8K split `div8k1k`, where the real image, not the round trip, is the 1.0) and the native-route baseline for all three controllers on subset500; both land in `BASELINES.md` (Sections F2 and A) automatically.

## Results (dev-200, OminiControl canny, seed 0)

<!-- RESULTS:BEGIN -->
_Last run: 2026-09-14. Regenerate with `bash scripts/run_all.sh`; this block is written by `scripts/04_fill_readme.py`._

| Decoder | n | Canny F1 @512 vs c512 (1 px) ↑ | Canny F1 @2048 vs c512 (4 px) ↑ | Canny F1 @2048 vs c2048 (1 px) ↑ | strict F1 @512 (anchor) | LPIPS ↓ | PSNR ↑ | SSIM ↑ | MUSIQ @512 ↑ | MUSIQ @2048 ↑ | s/img ↓ | peak GB ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real image (512; 2048 = bicubic x4 ‡) | 200 | 1.0000 | 0.5511‡ | 0.3095‡ | 1.0000 | 0.0000 | 80.00 | 1.0000 | 69.80 | 30.73 |  |  |
| VAE round trip (decode ceiling; 2048 = bicubic x4 ‡) | 200 | 0.9427 | 0.5338‡ | 0.3002‡ | 0.7574 | 0.0180 | 32.67 | 0.9282 | 69.84 | 30.66 |  |  |
| PiD round trip, student (generative ceiling; 2048 native) | 200 | 0.9162 | 0.7934 | 0.7142 | 0.6475 | 0.0517 | 28.83 | 0.8636 | 70.41 | 58.99 |  |  |
| PiD round trip, teacher | 25 | 0.9220 | 0.7855 | 0.6207 | 0.6595 | 0.0457 | 28.79 | 0.8910 | 75.03 | 59.30 |  |  |
| OminiControl + VAE decode (512 native; 2048 = bicubic x4 ‡, the interpolation route) | 200 | 0.7670 | 0.2855‡ | 0.1201‡ | 0.3647 | 0.5142 | 11.90 | 0.4394 | 70.40 | 34.08 | 0.03 | 34.5 |
| OminiControl GENERATING at 2048 + VAE decode (the native route) | 200 | 0.0371 | 0.0525 | 0.0246 | 0.0087 | 0.7622 | 6.26 | 0.0635 | 34.90 | 24.51 | 173.74 | 44.0 |
| OminiControl + vanilla PiD student, final latent (28/28) | 200 | 0.7781 | 0.7253 | 0.4954 | 0.3624 | 0.5092 | 11.81 | 0.4284 | 71.57 | 61.05 | 0.96 | 23.6 |
| OminiControl + vanilla PiD student, K=24 **(gate row)** | 200 | 0.7133 | 0.6620 | 0.4711 | 0.3005 | 0.5265 | 11.76 | 0.3999 | 72.42 | 61.95 | 0.95 | 23.6 |
| OminiControl + vanilla PiD student, K=16 | 200 | 0.5378 | 0.5562 | 0.3946 | 0.1916 | 0.6002 | 11.67 | 0.3480 | 74.79 | 66.92 | 0.96 | 23.6 |
| OminiControl + vanilla PiD teacher, final latent (28/28) | 200 | 0.7636 | 0.7114 | 0.4799 | 0.3587 | 0.4971 | 11.80 | 0.4352 | 71.94 | 58.84 | 17.53 | 21.5 |
| OminiControl + vanilla PiD teacher, K=24 **(gate row for teacher-based CGD)** | 200 | 0.6938 | 0.6654 | 0.4420 | 0.2945 | 0.5149 | 11.82 | 0.4153 | 72.02 | 60.44 | 17.54 | 21.5 |
| OminiControl + vanilla PiD teacher, K=16 | 200 | 0.5194 | 0.5296 | 0.3557 | 0.1868 | 0.5910 | 11.63 | 0.3686 | 72.28 | 62.42 | 17.52 | 21.5 |

Generation at 512 (FLUX.1-dev + OminiControl canny LoRA, 28 steps, seed 0): 5.11 s/img, shared by every row except the native-route row, whose s/img includes its own 2048 generation. ‡ = bicubic x4 upsample of a 512 output (interpolation route, not a native output). Native condition = Canny of the PiD round trip at 2048, decoded at the HELD-OUT decoder seed 7 (`bench/build_native_conditions.py --src-name pid_roundtrip_s7`; BENCHMARK v1.16). No evaluated row uses seed 7, so no row shares sampler noise with its own reference; the seed-0 round-trip row therefore scores about 0.71 there, not 1.0, and that value is the sampler-noise floor of the column.
<!-- RESULTS:END -->

How to read it (final protocol, 2026-09-12; every row has a 512 and a 2048 column):
- **Rows.** Reference rows (Real image, VAE round trip = decode ceiling, PiD round trip = generative ceiling) come from the paper benchmark's materialization of the same 200 images. The VAE decode row is 512-native; its 2048 values are the **interpolation route** (bicubic x4, marked ‡). The row "OminiControl GENERATING at 2048" is the **native route**: FLUX + the canny LoRA run at 2048 directly, then the VAE decode (4 MP, outside FLUX's training range; ~2.5 min and 44 GB per image). **Result (2026-09-12): this route COLLAPSES at 2048.** The outputs are near-black textures that faintly trace the condition (MUSIQ 35, canny F1 0.04, PSNR 6); the same script at 1024 gives a normal image (`outputs/omini_vae_gen1024_512/000.png`), so the failure is the 4 MP setting with a 512-trained token-concat LoRA, not the pipeline. Read that row as "the native route is not available for this controller at 2048", not as a competitor; the 2048 numbers to beat are the PiD rows. The PiD rows are the **pixel-decoder route** from the 512 latent: student = the released 4-step distilled decoder, teacher = the undistilled v1.5 decoder (25 steps, CFG 5) that CGD is trained on. The three routes to 2048 are now distinguishable in one table.
- **Canny F1 @2048 vs c2048** scores the 2048 output against the 2048 edge map c2048 (Canny of the PiD round trip of the real image, tolerance 1 px); since BENCHMARK v1.16 the map comes from the round trip at held-out decoder seed 7, so the seed-0 round trip scores 0.708 there (the column's sampler-noise floor) and only the seed-7 round trip is 1.0. This is the native-condition setting in which a CGD variant may receive the 2048 map (`update.md` Section 7).
- **Canny F1 @512 (matched)** is the number to optimize. Outputs are compared at the VAE's native 512 (PiD/CGD outputs are downsampled from 2048 with `cv2.INTER_AREA`), so a gain cannot come from having more pixels. F1 between `cv2.Canny(gray(output), 100, 200)` and the input condition with a matching tolerance of **one condition pixel** (1 px at 512; BENCHMARK v1.8, the paper metric). The **strict** pixel-exact F1 of the original loop is kept as the 512 anchor column (`canny_f1s_512`); its ranking agrees with the tolerant one at 512.
- **Canny F1 @2048 (native)** is the target: scored at PiD's native output against the condition resampled to 2048 (nearest), with the tolerance scaled to one condition pixel = **4 px at 2048**. The drop from @512 to @2048 within a method is the *resolution gap*: whether the native-resolution detail stayed condition-consistent. **Do not read the strict @2048 column as adherence**: the nearest-resampled reference is 4 px thick while re-extracted edges are 1 to 2 px, so pixel-exact F1 there measures the mismatch (a bicubic x4 upsample of the REAL image scores 0.16 strict); it is reported only for continuity with earlier runs. Canny thresholds are NOT rescaled with resolution, so the native column also requires sharp native-scale edges: the same bicubic x4 real image scores 0.55 tolerant at 2048 (recall loss, edge density 9.5% -> 1.4%), which is intended (a decoder must synthesize the detail, not blur). For the VAE row, which is 512-native, the @2048 value is a **reference** computed on a bicubic x4 upsample of its output (marked "bicubic x4 ref."; the paper marks such cells with a double dagger): the interpolation route to 2048, the level reached with that content and no native detail. The benchmark's never-upsample rule applies to claims, not to this reference; the upsampled REAL image scores 0.55 under it, against 0.80 for PiD's round trip of the same latent.
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

**Paths that are ours, not yours.** Two more variables point at data this repo does not ship, and both default to the path they
have on our group server, so set them if you are elsewhere:

| variable | what it points at | used by | needed for the quick start? |
|---|---|---|---|
| `CGD_BENCH_ROOT` | the materialized paper benchmark (round trips, `canny2048`, cached decodes) | `bench/common.py`, `scripts/03_score.py` | no — the three reference rows are simply omitted, with a note, and your own decoder rows are still scored |
| `CGD_RAW_ROOT` | downloaded raw datasets (MultiGen, ADE20K, DIV8K, DreamBench) | `bench/common.py` | no — only the `bench/build_*.py` set builders use it |

The quick start needs neither: it rebuilds the 200 images from the HF dataset above and scores what you generate. The full
paper benchmark under `bench/` does, and those sets are built on our server; ask the maintainer for the shared path.

Two further places carry our absolute paths on purpose. Every record in `results/` stores the `gen_dir` it was produced from,
which is provenance rather than configuration and is never read back. `bench/run/*.sh` are our server orchestration scripts,
hardcoded to the group container; read them for the exact commands, but do not expect them to run elsewhere.

External code (cloned, not vendored):
```bash
cd $CGD_ROOT
git clone --depth 1 https://github.com/Yuanshi9815/OminiControl.git
git clone --depth 1 https://github.com/nv-tlabs/PiD.git
bash cgd-dev200/scripts/env_pid.sh      # PiD's python deps (once)
```
Note: PiD pins `diffusers==0.37.1`, `transformers==4.57.1`, `numpy==1.26.4`; `env_pid.sh` installs them and therefore changes your environment's versions (see "Caveats"). Generation and scoring are verified under these versions.

Eval data (downloaded once; not redistributed). `scripts/00_select_dev200.py` rebuilds the 200 images and their canny
conditions from this dataset, so the quick start cannot run without it:
```
hf download limingcv/MultiGen-20M_canny_eval --repo-type dataset --local-dir $CGD_ROOT/data/multigen_canny_eval
```
The 200 rows are then selected by `sha256` from `bench/multigen5k/manifest.csv`, so you can verify you have exactly the same
images as us regardless of which revision you pulled; step 00 fails loudly if a sha does not match. Point
`MULTIGEN_DIR` at the `data/` subdirectory if you put it elsewhere.

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
