# CGD Dev Setting (internal, for method exploration)

**Purpose:** iterate on the method fast and pick the best variant. This is NOT the paper benchmark. Numbers from this setting are never reported; the paper uses `BENCHMARK_v1.md` only. Everything here is chosen for turnaround time and signal clarity.

## Why canny, and why so few images

- **Condition = canny edge.** Zero-model everywhere: the condition is `cv2.Canny(img, 100, 200)` on the source image (no DPT, no Mask2Former, no adapter decision pending), the OminiControl canny LoRA is released, and the scorer is `cv2.Canny` + F1 (CPU, deterministic, instant). Edges are also the most decode-sensitive structure we have (thin lines and boundaries blur through a lossy decode), so the effect we are optimizing is loudest here.
- **200 images.** Mean canny F1 and the per-image quality metrics are stable at 200; FID is not (needs thousands) and is dropped from the dev loop. 200 images decode in minutes per decoder variant.
- **Latents cached once.** The generator (28 FLUX steps) is the slow part; the decoder is fast. Generate the 200 conditioned latents ONCE and freeze them. Every method variant only runs the decoder on the identical latents, so comparisons are paired and low-variance.

## The fixed dev set

| Item | Value |
|---|---|
| Dataset | MultiGen-20M, canny split, **200 images**, fixed id list `dev/multigen_canny_dev200.txt` (random with seed 0, then frozen; never change it) |
| Condition | `cv2.Canny(src, 100, 200)` from the source image |
| Prompt | the MultiGen caption |
| Generator | FLUX.1-dev, OminiControl canny LoRA, **28 steps, guidance 3.5, seed 0**, generated at **512** (latent 64x64) |
| Cached latents | `dev/latents_canny_dev200_seed0/<controller>/` (the 200 conditioned latents per controller; every variant decodes THESE). Dev loop runs on **OminiControl** first; add `easycontrol/`, `fluxcn/` caches (200 latents each, cheap; OminiControl2 dropped in BENCHMARK v1.6) for the L2 modularity check |
| Reference | the real MultiGen source image (paired reference for LPIPS/PSNR) |

## Decoders

Two fixed reference rows, computed once and pinned:
1. **VAE decode** -> 512 output (condition-blind, deterministic)
2. **vanilla PiD** -> 2048 output (condition-blind, generative; = CGD minus the condition)

Both generative rows exist for three latents per image, all in the cache: `x_0` (28/28), `x_t@24`, `x_t@16`. **BENCHMARK v1.5 (2026-09-09): the MAIN setting is truncated (K = 16 or 24 of 28, undecided).** A CGD variant is therefore judged primarily at K, against vanilla PiD decoded from the SAME `x_t@K`, with `x_0` as the ablation. Run both K until the choice is made.

Every candidate is a **CGD variant -> 2048 output**, decoding the same cached latents. The conditioned-VAE archetype (Group B) is optional here; add it only if a variant needs the 2x2 to be interpreted.

Resolution (BENCHMARK v1.1): latents are generated at 512. VAE decode -> 512; PiD/CGD -> 2048 native. Score canny F1 in TWO views: (a) **matched 512**, downsampling the 2048 outputs with `cv2.INTER_AREA` (apples-to-apples with the VAE row; the primary dev signal), and (b) **native 2048** (the target). The within-method gap, (a) minus (b), tells you whether the native-resolution detail stayed condition-consistent.

## What to report per experiment (4 numbers, one line)

| Metric | Role | Tool |
|---|---|---|
| **Canny F1** (up), at matched 512 AND at native 2048 | the adherence signal; the matched-512 number is what we optimize, the 2048 number is the target | `cv2.Canny(out,100,200)` vs input condition (resampled to the scoring resolution), F1 |
| **LPIPS** (down) and PSNR (up) vs the real reference | fidelity: did we stay faithful to the photo, or hallucinate | `pyiqa lpips`, `psnr` |
| **MUSIQ** (up) | no-reference quality: did adherence cost image quality | `pyiqa musiq` |
| **Latency** (ms / image) | cost, especially for training-free variants | logged |

Not in the dev loop: FID / pFID / CLIP-FID (too noisy at 200), MANIQA (redundant with MUSIQ here), depth / seg / bbox / subject (promotion only).

## The gate (when does a variant "win")

A CGD variant is promoted only if, on the same 200 cached latents and at the SAME truncation point K as the vanilla-PiD row it is compared with:
- **matched-512 Canny F1 > vanilla PiD's matched-512 F1**, and **native-2048 F1 >= vanilla PiD's native-2048 F1**, and
- **MUSIQ >= vanilla PiD's MUSIQ** (within noise) and **LPIPS not worse** than vanilla PiD.

Beating VAE decode is necessary but not sufficient; the bar is vanilla PiD, because that isolates the value of re-injecting the condition. The primary gate is at the main truncation point K (16 or 24, undecided); the `x_0` gate is the ablation.

## Promotion ladder

| Level | Set | Purpose | Cost |
|---|---|---|---|
| L0 smoke | 20 images, canny | does it run, is it obviously broken | seconds |
| **L1 dev (the loop)** | **200 images, canny, VAE@512 vs PiD/CGD@2048** | **method selection; iterate here** | minutes |
| L2 confirm | same 200 + **depth dev200** (DPT MSE/RMSE) + the same 200 canny latents under the **other three controllers** | the winning variant does not regress on smooth structure, and its gain over vanilla PiD holds under every controller (modularity) | ~20 min |
| L3 benchmark | full `BENCHMARK_v1.md` | paper numbers | hours to days |

Depth is the second dev condition because it is the other released OminiControl spatial LoRA and it probes the opposite regime (smooth, low-frequency structure). A method that wins on canny but loses on depth is over-sharpening, not improving adherence.

## Rules

1. Never change the 200-image id list or the seed; if you must, it is a new dev set with a new name.
2. Never report dev-set numbers in the paper.
3. If a variant is trained (condition-adapter), its training data must be disjoint from the MultiGen evaluation split.
4. Compare only against the pinned VAE / vanilla PiD rows computed on the same cached latents; do not regenerate latents per experiment.
5. Log every run as one line: `variant, F1, LPIPS, PSNR, MUSIQ, ms/img, notes`. A shared CSV is enough.

## Parallel track (does not block the dev loop)

**Adapter training (BENCHMARK v1.3, Section 5b).** Three LoRAs (BENCHMARK v1.6) on spare GPUs while the canny loop iterates: bbox for OminiControl / EasyControl, seg for OminiControl. Official recipes, COCO train / ADE20K train, one pinned rendered-condition format (filled class-color boxes on black; ADE20K-palette masks). When they land, add `seg` and `bbox` dev200 caches per controller for L2/L3. Nothing in L0/L1 waits on this.

## Implementation and baseline results (2026-09-08, re-run from one generation)

**Repo (public):** https://github.com/wookiegen/cgd-dev200 (`scripts/00`-`04`, `dev200/manifest.csv`, `results/dev200_summary.md`; data/latents/outputs are rebuilt, not committed). On the group server the materialized set, cached latents (`x_0`, `x_t@16`, `x_t@24`), and outputs live at `/data/wookiekim/cgd/cgd-dev200/` next to clean clones of OminiControl and PiD in `/data/wookiekim/cgd/`.

**Baselines on the frozen dev-200 (OminiControl canny, seed 0, all rows from ONE generation, paired on all 200):**

| Decoder | n | Canny F1 @512 (matched) ↑ | Canny F1 @2048 (native; VAE row via bilinear x4) ↑ | LPIPS ↓ | PSNR ↑ | SSIM ↑ | MUSIQ @512 (matched) ↑ | MUSIQ (native) ↑ | decode s/img ↓ | peak mem GB ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| OminiControl + VAE decode (512, native) | 200 | 0.3647 | 0.0263 (bilinear x4) | 0.5142 | 11.90 | 0.4394 | 70.40 | 70.40 | 0.033 | 34.5 |
| OminiControl + vanilla PiD (final latent, 2048) | 200 | 0.3624 | 0.2217 | 0.5092 | 11.81 | 0.4284 | 71.57 | 61.05 | 0.959 | 23.6 |
| OminiControl + vanilla PiD, early-terminated at 24/28 (2048) | 200 | 0.3005 | 0.2033 | 0.5265 | 11.76 | 0.3999 | 72.42 | 61.95 | 0.953 | 23.6 |
| OminiControl + vanilla PiD, early-terminated at 16/28 (2048) | 200 | 0.1916 | 0.1622 | 0.6002 | 11.67 | 0.3480 | 74.79 | 66.92 | 0.961 | 23.6 |

Generation (FLUX.1-dev + OminiControl canny LoRA, 28 steps @512, seed 0): 5.11 s/img, shared by every row.

Readouts: (1) vanilla PiD does NOT beat the VAE on matched-512 canny F1 (paired mean delta -0.0024, PiD wins 74/200), so a condition-blind generative decoder buys no adherence; (2) PiD's within-method gap F1@512 - F1@2048 = +0.141, the resolution drift CGD must close; (3) the VAE row's @2048 value is a bilinear x4 REFERENCE (blur kills Canny response), showing the generative decoder synthesizes real edge content at native res; (4) early termination costs adherence monotonically (24/28 then 16/28). LPIPS/PSNR/SSIM are whole-image vs source and indicative only (edge-conditioned generation, not reconstruction). MUSIQ is higher-better; compare it only within a resolution (the @512 column across rows). Latents are NOT bit-reproducible across runs (max |d| ~0.4 on a std-0.69 latent), which is why the cache is shared and every decoder must read it.

**Gate, concretely (per truncation point, against the vanilla-PiD row decoded from the same latent):**

| Latent handed to the decoder | F1 @512 must exceed | F1 @2048 must exceed | MUSIQ @512 at least | LPIPS at most |
|---|---|---|---|---|
| `x_t@16` (16/28) | 0.1916 | 0.1622 | 74.8 | 0.600 |
| `x_t@24` (24/28) | 0.3005 | 0.2033 | 72.4 | 0.527 |
| `x_0` (28/28, ablation) | 0.3624 | 0.2217 | 71.6 | 0.509 |

The main setting is one of the two truncated rows (K = 16 or 24, BENCHMARK v1.5, undecided); report all three.
