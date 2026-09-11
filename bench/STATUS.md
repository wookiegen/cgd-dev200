# STATUS.md — where the benchmark stands (2026-09-10)

What exists on the group server, what is running unattended, what remains. Paths are under `/data/wookiekim/cgd/data/bench/` (`$CGD_BENCH_ROOT`) unless noted; results are in this repo under `results/bench/`; logs and completion markers in `/data/wookiekim/cgd/data/_logs/`. Raw data and model weights: `DATA.md`. The paper's tables are filled from `results/bench/SUMMARY.md` (regenerate with `python bench/assemble.py`).

## 1. Generated data on the server

| What | Path | State |
|---|---|---|
| Clean latents of the real crops (decoder-only reference rows) | `latents/ref/<split>/<id>.pt` | 12,000 done (multigen5k, ade20k_val2k, coco_val5k) |
| VAE round trip (512) | `outputs/ref/vae_roundtrip/<split>/<id>.png` | done |
| vanilla PiD round trip (2048 + 512 view) | `outputs/ref/pid_roundtrip/<split>/`, `outputs/ref/pid_roundtrip_512/<split>/` | done |
| OminiControl latents (x_0 + x_t at 16, 24) and its VAE decode | `latents/omini/<canny,depth>/<id>.pt`, `outputs/omini/<cond>/vae@28/` | RUNNING (multigen5k, 2 shards per condition) |
| vanilla PiD decodes of the OminiControl latents at K = 28 / 24 / 16 | `outputs/omini/<cond>/pid@<K>/`, `pid@<K>_512/` | queued (`run/run_omini_block.sh`) |
| EasyControl latents (canny, depth; seg on ade20k_val2k) and FLUX ControlNet latents (canny, depth) | `latents/<easycontrol,fluxcn>/<cond>/`, `outputs/<ctrl>/<cond>/vae@28/` | queued (`run/run_queue_controllers.sh`, starts when the omini decodes finish) |
| their PiD decodes at K | `outputs/<ctrl>/<cond>/pid@<K>/` | queued (`run/run_block.sh <ctrl>`) |
| CGD training triplets | `train/<set>/` | NOT built (`make_targets.py`; ~12 GPU-hours; waits for a slot after the baselines) |

Completion markers: `done_ref_all.txt`, `done_latents_omini_all.txt`, `done_omini_block.txt`, `done_latents_<ctrl>_<cond>.txt`, `done_easycontrol_block.txt`, `done_fluxcn_block.txt`, `done_queue.txt`. Status lines: `omini_block_status.txt`, `queue_status.txt`, `<ctrl>_block_status.txt`.

## 2. Results in the repo (`results/bench/`)

| File | Content |
|---|---|
| `<split>/[<controller>.]<method>@<res>[.<metrics>][.<condition>].json` | one record per condition: adherence + quality (schema: paper repo `SCORING_HARNESS.md`). The controller prefix (e.g. `omini.vae@512.canny.json`) keeps the three controller blocks apart; `.<metrics>` marks metric-subset runs (e.g. `.vlm`); `.<condition>` marks runs restricted to one condition of the split. `assemble.py` merges everything by (split, controller, method, res, condition) |
| `<split>/<method>@<res>_per_image.csv` | per-sample scores (paired analysis, figures) |
| `efficiency.json` | tab:efficiency measurements (one H200) |
| `SUMMARY.md`, `summary.csv` | everything merged, one row per (split, condition, method, controller, res) |

Done: `real`, `vae_roundtrip`, `pid_roundtrip` on all three splits at 512 (and 2048 for PiD), incl. all eight no-reference metrics on multigen5k. Method labels for the controller blocks: `vae` (the controller's VAE decode, `--controller <ctrl>`), `pid_k28`, `pid_k24`, `pid_k16`.

## 3. Findings so far

- **Edge metric changed to tolerant F1 (BENCHMARK v1.8, 2026-09-11)**: strict F1 at 2048 compared 1-2 px re-extracted edges with the 4-px-thick nearest-resampled reference (a bicubic x4 of the REAL image scores 0.149 strict), so the strict 0.64 -> 0.26 drop of the PiD round trip was mostly the metric. Paper metric = F1 with a one-condition-pixel tolerance (1 px @512, 4 px @2048); strict kept as `f1_strict`. Tolerant: VAE round trip 0.94 @512; PiD round trip 0.92 -> 0.79; OminiControl + VAE decode 0.76; + PiD K=28 0.78 -> 0.73; K=16 0.54 -> 0.56. `bench/rescore_canny.py` patched the existing records.
- Deterministic decode ceiling for strict canny F1 at 512: 0.747 (VAE round trip; 0.94 tolerant); segmentation and layout are unchanged by the decode (53.6 / 72.1 vs 53.7 / 72.7 on the real images). Depth / seg / layout scorers resize internally, so their 2048 values are flat by construction (moved to a supplementary table).
- vanilla PiD on the same clean latent: canny 0.637 at the 512 view and 0.256 at native 2048 (the resolution gap), depth MSE 62 vs 11 for the VAE, seg / layout unchanged, MANIQA higher (0.58 vs 0.48): the blind generative decoder invents fine structure.
- FLUX + PiD reaches 2048 in 2.4 to 3.3 s vs 36 s for native FLUX + VAE; native FLUX at 4096 is out of memory on an H200, FLUX-at-1024 + PiD takes 12 to 15 s.
- 512-generation behavior of EasyControl and FLUX ControlNet Union-Pro-2.0 verified visually (both follow the condition; EasyControl's seg LoRA reads the ADE20K palette render).

## 4. Remaining work (method-independent unless stated)

| # | Task | Needs | Fills |
|---|---|---|---|
| 1 | Let the queue finish; `assemble.py`; fill the three controller blocks and the truncation sweep {28, 24, 16} | running | tab:main, tab:noref, tab:recon controller rows |
| 2 | Train the 3 LoRAs (seg: OminiControl; bbox: OminiControl, EasyControl) with the official recipes on `train/ade20k_train` and `train/coco_train` (drop `blocked`, use `split`), then their latent caches + blocks | ~1 GPU-day each | seg / layout cells of the controller blocks |
| 3 | Conditioned VAE decoder (the 2x2 archetype): train on the OminiControl latents + conditions, score with the harness | small training job | tab:decoder-conditioning |
| 4 | CGD training triplets (`make_targets.py`, full COCO train) | ~12 GPU-hours | adapter training data |
| 5 | Condition-scale sweep for fig:ceiling on dev-200 (`condition_scale` in OminiControl's `generate`) | ~1 GPU-hour | fig:ceiling |
| 6 | Eyeball the lifted numbers of tab:contextual / tab:subject against the official PDFs | reading | supplementary |
| 7 | **CGD rows** (every table) and the choice of K (16 or 24; the truncation sweep informs it) | THE METHOD | |
| 8 | Subject track (DreamBench, released subject LoRAs) | low priority | tab:subject |
| 9 | 1024 side comparison (QUEUED, not launched): EasyControl + FLUX ControlNet generated at 1024, VAE-decoded, downsampled to 512, scored on `subset500` (tests the 512-resolution caveat; side comparison only) | ~4 GPU-hours, after the blocks | supplementary table |
| 10 | Checkpoint consistency: vanilla PiD rows use the 2K checkpoint, a trained CGD will be v1.5-based (only undistilled FLUX checkpoint); re-decode vanilla rows with v1.5 distilled (~8 GPU-hours per block) or keep 2K as headline; decide with the method | decision | tab:main PiD rows |

Rules that bit us: never let two generator processes write the same set without disjoint shards (a race produced six corrupt PNGs, since regenerated); every decoder reads the cached latents, never regenerates; the harness refuses to upsample.
