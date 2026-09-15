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

Beating VAE decode is necessary but not sufficient; the bar is vanilla PiD, because that isolates the value of re-injecting the condition. The primary gate is at the main truncation point K (K = 16 (BENCHMARK v1.17, 2026-09-15)); the `x_0` gate is the ablation.

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

**Adapter training (BENCHMARK v1.3, Section 5b; reduced in v1.9).** One OPTIONAL LoRA on spare GPUs while the canny loop iterates: seg for OminiControl (the bbox LoRAs were dropped with the layout condition on 2026-09-12; EasyControl's released seg LoRA stands in meanwhile). Official recipe, ADE20K train, the pinned ADE20K-palette mask render. If it lands, add a `seg` dev200 cache for OminiControl for L2/L3. Nothing in L0/L1 waits on this.

## Implementation and baseline results (2026-09-08, re-run from one generation)

**Repo (public):** https://github.com/wookiegen/cgd-dev200 (`scripts/00`-`04`, `dev200/manifest.csv`, `results/dev200_summary.md`; data/latents/outputs are rebuilt, not committed). On the group server the materialized set, cached latents (`x_0`, `x_t@16`, `x_t@24`), and outputs live at `/data/wookiekim/cgd/cgd-dev200/` next to clean clones of OminiControl and PiD in `/data/wookiekim/cgd/`.

**FINAL PROTOCOL (2026-09-12 evening; `scripts/03_score.py` rewritten):** the dev-200 reference table now has a 512 and a 2048 column for EVERY row in the tolerant F1, a "2048 vs native condition" column (Canny of the PiD round trip at 2048, 1 px), the strict F1 at 512 as the anchor, and three routes to 2048 that are distinguishable: interpolation (bicubic x4 of the 512 output, ‡), the native route (OminiControl GENERATING at 2048 + VAE, `scripts/01b_generate_omini_2048.py`), and the pixel-decoder route (PiD student = released 4-step; PiD teacher = undistilled v1.5, 25 steps CFG 5, `scripts/02b_decode_teacher.py`), plus reference rows (Real, VAE round trip, PiD round trip) mapped from the paper benchmark. **The live numbers are the auto-filled table in the repo README (`scripts/04_fill_readme.py`) and `BASELINES.md` Section E; the table below is the 2026-09-12 snapshot for the gate.**

**Baselines on the frozen dev-200 (OminiControl canny, seed 0, all rows from ONE generation, paired on all 200):**

| Decoder | n | Canny F1 @512 (matched, tolerant) ↑ | Canny F1 @2048 (tolerant 4 px) ↑ | Canny F1 @2048 vs NATIVE condition (1 px) ↑ **[EVERY CELL IN THIS COLUMN IS SUPERSEDED BY v1.16; read `results/dev200_summary.md`]** | strict F1 @512 (anchor) | LPIPS ↓ | PSNR ↑ | SSIM ↑ | MUSIQ @512 ↑ | MUSIQ @2048 ↑ | s/img ↓ | peak GB ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real image (512; 2048 = bicubic x4 ‡) | 200 | 1.0000 | 0.5511‡ | 0.3105‡ | 1.0000 | 0.0000 | 80.00 | 1.0000 | 69.80 | 30.73 |  |  |
| VAE round trip (decode ceiling; 2048 = bicubic x4 ‡) | 200 | 0.9427 | 0.5338‡ | 0.3013‡ | 0.7574 | 0.0180 | 32.67 | 0.9282 | 69.84 | 30.66 |  |  |
| PiD round trip, student (generative ceiling; 2048 native) | 200 | 0.9162 | 0.7934 | 1.0000 | 0.6475 | 0.0517 | 28.83 | 0.8636 | 70.41 | 58.99 |  |  |
| OminiControl + VAE decode (512 native; 2048 = bicubic x4 ‡, the interpolation route) | 200 | 0.7670 | 0.2855‡ | 0.1211‡ | 0.3647 | 0.5142 | 11.90 | 0.4394 | 70.40 | 34.08 | 0.03 | 34.5 |
| OminiControl GENERATING at 2048 + VAE decode (the native route; COLLAPSED, see note) | 200 | 0.0371 | 0.0525 | 0.0245 | 0.0087 | 0.7622 | 6.26 | 0.0635 | 34.90 | 24.51 | 174.42 | 44.0 |
| OminiControl + vanilla PiD student, final latent (28/28) | 200 | 0.7781 | 0.7253 | 0.5316 | 0.3624 | 0.5092 | 11.81 | 0.4284 | 71.57 | 61.05 | 0.96 | 23.6 |
| OminiControl + vanilla PiD student, K=24 **(gate row)** | 200 | 0.7133 | 0.6620 | 0.4992 | 0.3005 | 0.5265 | 11.76 | 0.3999 | 72.42 | 61.95 | 0.95 | 23.6 |
| OminiControl + vanilla PiD student, K=16 | 200 | 0.5378 | 0.5562 | 0.4154 | 0.1916 | 0.6002 | 11.67 | 0.3480 | 74.79 | 66.92 | 0.96 | 23.6 |
| OminiControl + vanilla PiD teacher, final latent (28/28) | 200 | 0.7636 | 0.7114 | 0.4807 | 0.3587 | 0.4971 | 11.80 | 0.4352 | 71.94 | 58.84 | 17.53 | 21.5 |
| OminiControl + vanilla PiD teacher, K=24 **(gate row for teacher-based CGD)** | 200 | 0.6938 | 0.6654 | 0.4427 | 0.2945 | 0.5149 | 11.82 | 0.4153 | 72.02 | 60.44 | 17.54 | 21.5 |
| OminiControl + vanilla PiD teacher, K=16 | 200 | 0.5194 | 0.5296 | 0.3562 | 0.1868 | 0.5910 | 11.63 | 0.3686 | 72.28 | 62.42 | 17.52 | 21.5 |

(Rescored 2026-09-12 under the final protocol on the SAME outputs; teacher + native-route rows landed 13:29 KST; live table = repo README / `BASELINES.md` Section E.) **Native-route note:** OminiControl generating at 2048 directly COLLAPSES (near-black textures tracing the condition; MUSIQ 35, F1 0.04), while the same script at 1024 gives a normal image, so the row records that the route is unavailable for this controller at 2048; the subset500 native-route baseline with all three controllers is queued. **Teacher gate (for a CGD built on the undistilled teacher):** K=24 > 0.6938 @512 and > 0.6654 @2048 (vs c512); K=16 0.5194 / 0.5296; final latent 0.7636 / 0.7114; MUSIQ / LPIPS not worse than the teacher row. The teacher sits slightly below the student in adherence at every K and costs 17.5 s per decode.

Generation (FLUX.1-dev + OminiControl canny LoRA, 28 steps @512, seed 0): 5.11 s/img, shared by every row.

Readouts: (1) vanilla PiD does NOT beat the VAE on matched-512 canny F1 (paired mean delta -0.0024, PiD wins 74/200), so a condition-blind generative decoder buys no adherence; (2) PiD's within-method gap F1@512 - F1@2048 = +0.141, the resolution drift CGD must close; (3) the VAE row's @2048 value is a bilinear x4 REFERENCE (blur kills Canny response), showing the generative decoder synthesizes real edge content at native res; (4) early termination costs adherence monotonically (24/28 then 16/28). LPIPS/PSNR/SSIM are whole-image vs source and indicative only (edge-conditioned generation, not reconstruction). MUSIQ is higher-better; compare it only within a resolution (the @512 column across rows). Latents are NOT bit-reproducible across runs (max |d| ~0.4 on a std-0.69 latent), which is why the cache is shared and every decoder must read it.

**Gate, concretely (per truncation point, against the vanilla-PiD row decoded from the same latent):**

| Latent handed to the decoder | tolerant F1 @512 must exceed | tolerant F1 @2048 must exceed | (strict F1 @512 anchor) | MUSIQ @512 at least | LPIPS at most |
|---|---|---|---|---|---|
| `x_t@16` (16/28) | 0.5378 | 0.5562 | 0.1916 | 74.8 | 0.600 |
| `x_t@24` (24/28) | 0.7133 | 0.6620 | 0.3005 | 72.4 | 0.527 |
| `x_0` (28/28, ablation) | 0.7781 | 0.7253 | 0.3624 | 71.6 | 0.509 |

The main setting is one of the two truncated rows (K = 16 or 24, BENCHMARK v1.5, undecided); report all three.

**Metric note (BENCHMARK v1.8; dev loop updated 2026-09-12):** the paper's canny metric is F1 with a ONE-CONDITION-PIXEL tolerance (1 px at 512, 4 px at 2048), and since 2026-09-12 the dev-200 gate is in that metric at BOTH resolutions (`scripts/03_score.py` reports `canny_f1_*` tolerant and `canny_f1s_*` strict; the README table carries both). The strict pixel-exact F1 stays as the 512 anchor (same ranking as the tolerant one there). The strict @2048 value is NOT a gate: the nearest-resampled reference is 4 px thick against 1 to 2 px re-extracted edges, so it measures the mismatch (a bicubic x4 of the REAL image scores 0.16 strict). Canny thresholds are not rescaled with resolution, so the tolerant native column still demands sharp native-scale edges (the same bicubic x4 real image scores 0.55 tolerant at 2048, edge density 9.5% -> 1.4%): a decoder has to synthesize the detail, not blur. Colleagues who cloned the first commit must `git pull` and re-run `03_score.py`; ids, latents, and outputs are unchanged. Paper text: EXPERIMENTS_draft.tex, "Scoring Protocol Across Resolutions".

## 2026-09-15: the truncation point is K = 16 (user decision; BENCHMARK v1.17)

The CGD gate is therefore read against the **K=16** vanilla-PiD row, not K=24. From the dev-200 table:
a CGD variant at K=16 must beat **0.5378** tolerant F1 at the 512 view and **0.5562** at native 2048 (vs c512),
with MUSIQ not lower and LPIPS not worse than that same row. The K=24 thresholds (0.7133 / 0.6620) and the x0
thresholds (0.7781 / 0.7253) remain valid for variants evaluated at those points but are no longer the headline gate.
For a CGD built on the undistilled teacher, the teacher K=16 row is the reference: 0.5194 / 0.5296.

Note on the third edge column: since BENCHMARK v1.16 the c2048 reference is a round trip at held-out decoder seed 7, so the
dev-200 PiD round-trip row reads 0.7142 there rather than 1.0, and that value is the column's sampler-noise floor.
