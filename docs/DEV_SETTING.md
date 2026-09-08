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
| Cached latents | `dev/latents_canny_dev200_seed0/<controller>/` (the 200 conditioned latents per controller; every variant decodes THESE). Dev loop runs on **OminiControl** first; add `omini2/`, `easycontrol/`, `fluxcn/` caches (200 latents each, cheap) for the L2 modularity check |
| Reference | the real MultiGen source image (paired reference for LPIPS/PSNR) |

## Decoders

Two fixed reference rows, computed once and pinned:
1. **VAE decode** -> 512 output (condition-blind, deterministic)
2. **vanilla PiD** -> 2048 output (condition-blind, generative; = CGD minus the condition)

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

A CGD variant is promoted only if, on the same 200 cached latents:
- **matched-512 Canny F1 > vanilla PiD's matched-512 F1**, and **native-2048 F1 >= vanilla PiD's native-2048 F1**, and
- **MUSIQ >= vanilla PiD's MUSIQ** (within noise) and **LPIPS not worse** than vanilla PiD.

Beating VAE decode is necessary but not sufficient; the bar is vanilla PiD, because that isolates the value of re-injecting the condition.

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

**Adapter training (BENCHMARK v1.3, Section 5b).** Five LoRAs on spare GPUs while the canny loop iterates: bbox for OminiControl / OminiControl2 / EasyControl, seg for OminiControl / OminiControl2. Official recipes, COCO train / ADE20K train, one pinned rendered-condition format (filled class-color boxes on black; ADE20K-palette masks). When they land, add `seg` and `bbox` dev200 caches per controller for L2/L3. Nothing in L0/L1 waits on this.
