# bench/ — frozen manifests of the CGD benchmark (bench_v1)

The paper benchmark is specified in the paper repo's `docs/BENCHMARK_v1.md` (v1.9; the bounding-box layout condition was dropped from the paper on 2026-09-12, so `coco_val5k` and `train/coco_train` below are kept but unused). This directory pins **which images**, in **which order**, with **which conditions and captions**, so that every controller, decoder, and truncation point is evaluated on identical rows. Images, conditions, labels, and latents are **rebuilt** from the public sources with the scripts here and are never committed (LAION-derived, COCO, and ADE20K images cannot be redistributed).

## Sets

| Set | Rows | Source | Conditions | Notes |
|---|---|---|---|---|
| `multigen5k` | 5000 | HF `limingcv/MultiGen-20M_canny_eval`, validation, shard order | canny (`cv2.Canny(gray,100,200)`), depth (`Intel/dpt-large`) | shared by canny and depth: the depth_eval split holds the same 5000 images in the same order (sha256-verified); all 5000 rows kept, 25 byte-identical duplicates marked `dup_group`; `dev200_idx` links to `../dev200/manifest.csv` |
| `ade20k_val2k` | 2000 | HF `limingcv/Captioned_ADE20K`, validation | seg (palette render of the GT label map) | captions = ControlNet++ `prompt`; label 0 = other/ignore, 1..150 classes; `palette.json` derived from the dataset's own `control_seg` |
| `coco_val5k` (unused since 2026-09-12) | 5000 | COCO 2017 val + instances/captions | bbox (filled class-color boxes on black, `class_colors.json`) | order = ascending image id; caption = lowest caption annotation id; `boxes.jsonl` lists kept boxes; layout metrics use `has_boxes = 1` rows, FID uses all |
| `dreambench750` | 750 | HF `google/dreambooth` | reference image | 30 subjects x 25 prompts; reference = first image per subject; `{0} {1}` -> class name |
| `train/multigen_train30` | ~36.8k | first 30 shards of HF `limingcv/MultiGen-20M_train` | canny/depth at train time | `blocked` column: 91 eval images were found in these shards |
| `train/ade20k_train` | 20210 | HF `limingcv/Captioned_ADE20K`, train | seg at train time | `blocked` column |
| `train/coco_train` (unused since 2026-09-12) | 118287 | COCO 2017 train | bbox at train time | `blocked` column |

Exact counts, manifest checksums, and the source revisions at build time are in `BENCH.json`. Where the raw data, materialized sets, and model weights live on the group server, and which dataset serves which condition: `DATA.md`. What has been generated and scored, what is running, and what remains: `STATUS.md`.

## Conventions

- **`sample_id`**: five digits, zero-padded, in source order (file, then row), so it is reproducible from the source alone. Latent caches and outputs are keyed by it: `latents/<controller>/<condition>/<sample_id>.pt`, `outputs/<controller>/<condition>/<decoder>@<K>/<sample_id>.png`.
- **`sha256`**: of the ORIGINAL image bytes (parquet bytes or the file on disk). Use it to verify your copy of a set.
- **Square crop** (non-square sources, i.e. ADE20K and COCO): shorter side -> 512 (image bicubic, label map nearest), center crop 512x512; masks and boxes receive the identical transform, boxes whose center falls outside the crop are dropped and the rest clipped; `crop_box = x0,y0,x1,y1` in resized coordinates. MultiGen images are already 512x512.
- **Conditions come from our pinned annotators** (`VERSION.json` in each set records the tool, version, thresholds, palette or color map); a dataset's shipped control maps are only cross-checks.
- **`subset500`**: a fixed 500-row subset per eval set (`numpy.random.default_rng(500).choice(n, 500, replace=False)`) for the later three-seed variance run.
- **Leakage blocklist**: `blocklist/eval_sha256.txt` is the union of every eval image's sha. Every training loader must drop rows whose sha is listed; the training manifests carry this as the `blocked` column. Finding at build time: 91 of the 5000 MultiGen eval images occur in the first 30 MultiGen train shards (188 rows).

## Rebuild

Sources are expected under `$CGD_RAW_ROOT` (default `/data/wookiekim/cgd/data`: `multigen_canny_eval/`, `captioned_ade20k/`, `coco2017/`, `dreambench/`, `multigen_train_subset/`), materialized sets are written under `$CGD_BENCH_ROOT` (default `/data/wookiekim/cgd/data/bench`), manifests into this directory.

```bash
cd bench
python build_multigen5k.py        # images + canny conditions + manifest
python depth_dpt.py --device cpu  # depth conditions (DPT-Large), ~1 h on CPU, minutes on a GPU
python build_ade20k_val2k.py      # cropped images, label maps, seg conditions, palette.json
python build_coco_val5k.py        # cropped images, bbox conditions, boxes.jsonl, class_colors.json
python build_dreambench750.py     # reference images + prompt list
python build_train_manifests.py   # blocklist + train manifests (blocked column)
python finalize_bench.py          # BENCH.json
```

Materialized layout per eval set: `images/<id>.png`, `conditions/<cond>/<id>.png` (depth also `conditions/depth_raw/<id>.npy`, float16, the scorer's reference), `labels/<id>.png` (ADE20K), `refs/<subject>.jpg` (DreamBench).

## Training triplets (`make_targets.py`)

The CGD adapter is trained per condition on its own training set (paper Method, "Training the Conditioned Decoder"). `make_targets.py --set <multigen_train30|ade20k_train|coco_train>` reads the training manifest, drops `blocked` rows, and writes under `$CGD_BENCH_ROOT/train/<set>/`: the 512 source crop (`images512/`), the clean FLUX latent (`latents/<id>.pt`, diffusers scaled space), the **2048 target** = vanilla PiD's four-step decode of that clean latent (`targets/<id>.jpg`, q95), and the conditions at 512: canny and depth extracted from the target's area-downsampled view (consistent with the target by construction), plus the GT-rendered seg (ADE20K) or bbox (COCO) condition after the same crop. The re-noised input latent is sampled in the training loop. Shard across GPUs with `--shard k --nshards n`; `--dry-run` validates parsing and GT rendering without models. Roughly 1 s per image on one H200 for the PiD decode.

## Scoring harness (`harness.py`, `scorers.py`)

`harness.py --method <label> --split <set> --gen-dir <dir of <id>.png> --res <512|2048>` scores any method's images with the frozen scorers of `scorers.py` (canny F1 with a one-condition-pixel tolerance, 1 px at 512 and 4 px at 2048, plus the strict pixel F1 as `f1_strict` for the published-protocol anchor; DPT-Large depth MSE/RMSE after per-image scale+shift alignment, in 0-255 units; Mask2Former ADE mIoU at dataset level; GroundingDINO layout success rate and mIoU under the MIGC protocol; pyiqa MUSIQ/NIQE/MANIQA/Q-Align; PSNR/SSIM/LPIPS/DISTS vs the real crop; clean-fid FID and patch-FID at the 512 view). A 2048-native directory is scored at `--res 512` through the pinned `INTER_AREA` downsample and at `--res 2048` natively; a 512-native method is refused at 2048 (never upsample). One JSON record per condition (schema in the paper repo's `SCORING_HARNESS.md`) plus a per-image CSV under `results/bench/<split>/`. The Real row is `--method real --gen-dir <set>/images`.

`make_reference_rows.py` produces the decoder-only reference rows (VAE round trip at 512, vanilla PiD round trip at 2048 + 512 view) from the real crops, caching the clean latent per sample; `measure_efficiency.py` fills the efficiency table (FLUX at the output resolution + VAE vs FLUX at a quarter + PiD, one GPU). `flux_pid.py` wraps the FLUX VAE and the PiD decoder in the shared latent convention.

Environment notes: pyiqa's Q-Align needs `pip install icecream` and, under torch < 2.6, the `q-future/one-align` `.bin` shards converted to safetensors inside the HF snapshot (transformers refuses `torch.load` there, and it ignores local safetensors when given the repo id, so `scorers.NoRef` redirects the loader to the local snapshot directory); both are done in the group container.

## Versioning

`bench_v1` is frozen with the paper's BENCHMARK v1.7. Any change to a set (rows, order, crop rule, annotator version, palette, color map) is a new version with a changelog line in `BENCH.json` and in the paper's `BENCHMARK_v1.md`; never edit a frozen manifest in place.
