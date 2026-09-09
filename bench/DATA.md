# DATA.md — what is downloaded, where it lives on the group server, and what it is for

Everything the CGD benchmark and the adapter training need was downloaded on 2026-09-09 and is shared on the group server. **Do not re-download**: point your environment at the paths below. Sizes are as measured; HF revisions are the commit at download time (`refs/main` in the cache, or `BENCH.json` for datasets).

## 0. Environment on the group server

| What | Value |
|---|---|
| Container | `wookiekim_tfso` (diffusers 0.37.1, transformers 4.57.1, numpy 1.26.4 = PiD's pins; pyiqa, clean-fid installed) |
| HF cache (models + datasets) | `/data/wookiekim/.cache/huggingface/hub` (the container sets `HF_HOME=/data/wookiekim/.cache/huggingface`) |
| Raw datasets | `/data/wookiekim/cgd/data/<name>/` (`$CGD_RAW_ROOT`, default in `bench/common.py`) |
| Materialized benchmark sets | `/data/wookiekim/cgd/data/bench/<set>/` (`$CGD_BENCH_ROOT`) |
| PiD repo + checkpoints | `/data/wookiekim/cgd/PiD` (`$PID_ROOT`), checkpoints under `checkpoints/` |
| OminiControl, EasyControl clones | `/data/wookiekim/cgd/OminiControl`, `/data/wookiekim/cgd/EasyControl` |
| Download / build logs | `/data/wookiekim/cgd/data/_logs/` |

From another container or the host, export `HF_HOME=/data/wookiekim/.cache/huggingface` (and `HF_HUB_OFFLINE=1` to be sure nothing is fetched) and `from_pretrained(<id>)` resolves from the shared cache. Files written by the container are root-owned: readable by everyone, writable only inside that container. Copy what you need to modify.

## 1. Conditions: which dataset, which role, which scorer

| Condition | Evaluation set (frozen manifest) | Training set (manifest, blocklist applied, 500-row `val` split) | Condition input | Frozen scorer |
|---|---|---|---|---|
| Canny edge | `multigen5k` = MultiGen-20M canny eval, 5000 | `train/multigen_train30` = 30 shards of MultiGen-20M train, 36,810 rows (188 blocked) | `cv2.Canny(gray, 100, 200)` at 512 | same Canny, F1 |
| Depth | `multigen5k` (the same 5000 images, verified) | same as canny (depth is extracted from any image) | `Intel/dpt-large` at 512 | DPT-Large, scale+shift aligned MSE / RMSE |
| Segmentation | `ade20k_val2k` = ADE20K val, 2000, GT masks, ControlNet++ captions | `train/ade20k_train` = ADE20K train, 20,210 rows (9 blocked) | GT label map rendered with `ade20k_val2k/palette.json` | `facebook/mask2former-swin-large-ade-semantic`, mIoU |
| Bbox layout | `coco_val5k` = COCO 2017 val, 5000 (4925 with boxes after the crop) | `train/coco_train` = COCO 2017 train, 118,287 rows | GT boxes rendered with `coco_val5k/class_colors.json` | `IDEA-Research/grounding-dino-base`, success rate @ IoU 0.5 and mIoU |
| Subject / reference | `dreambench750` = DreamBench, 30 subjects x 25 prompts | Subjects200K downloaded (no manifest yet; the subject track uses released LoRAs) | reference image | `facebook/dino-vits16` (DINO), `openai/clip-vit-large-patch14` (CLIP-I, CLIP-T) |

Non-square sources (ADE20K, COCO) are center-cropped to 512 after resizing the shorter side (BENCHMARK v1.7); masks and boxes get the identical transform. Every eval image's sha256 is in `blocklist/eval_sha256.txt`; training loaders must drop rows with `blocked = 1`.

## 2. Raw datasets (downloaded)

| Dataset | Source (HF id / URL) | Revision at download | Local path | Size | Content |
|---|---|---|---|---|---|
| MultiGen-20M canny eval | `limingcv/MultiGen-20M_canny_eval` | `fc7c740` | `/data/wookiekim/cgd/data/multigen_canny_eval/` | 2.1 GB | validation 5000 + test 500, 512x512, `image`, `text` |
| MultiGen-20M depth eval | `limingcv/MultiGen-20M_depth_eval` | `5ef53bd` | `/data/wookiekim/cgd/data/multigen_depth_eval/` | 2.1 GB | validation 5000 (same images/order as canny eval), `control_depth` kept as a cross-check only |
| MultiGen-20M train, first 30 shards | `limingcv/MultiGen-20M_train` (`data/train-0000[0-2]?-of-*.parquet`) | `451da2a` | `/data/wookiekim/cgd/data/multigen_train_subset/` | 15 GB | 36,810 rows; `EVAL_OVERLAP_sha256.txt` lists the 91 eval images found inside |
| Captioned ADE20K (ControlNet++) | `limingcv/Captioned_ADE20K` | `1406fc5` | `/data/wookiekim/cgd/data/captioned_ade20k/` | 6.6 GB | train 20,210 + validation 2,000: `image`, `prompt`, `detailed_prompt`, `seg_map`, `control_seg`, `image_path` |
| COCO 2017 | images.cocodataset.org (`val2017.zip`, `train2017.zip`, `annotations_trainval2017.zip`) | 2017 release | `/data/wookiekim/cgd/data/coco2017/{val2017,train2017,annotations,zips}` | 39 GB | 5000 val + 118,287 train images, instances + captions for both |
| DreamBench | `google/dreambooth` | `dc09c98` | `/data/wookiekim/cgd/data/dreambench/dataset/` | 108 MB | 30 subjects, `prompts_and_classes.txt` |
| Subjects200K | `Yuanshi/Subjects200K` | HF main 2026-09-09 | `/data/wookiekim/cgd/data/subjects200k/` | 9.9 GB | 32 train shards (OminiControl's subject training data) |

Not downloaded on purpose: MultiAspect-4K-1M (PiD's training corpus; the adapter trains on the per-condition sets above), OminiControl2 (no released weights; controller dropped), text-to-image-2M, COCOStuff.

## 3. Materialized benchmark sets (rebuilt from the raw data by the scripts in this directory)

| Set | Path | Size | Files |
|---|---|---|---|
| `multigen5k` | `/data/wookiekim/cgd/data/bench/multigen5k/` | 4.8 GB | `images/` 5000, `conditions/canny/` 5000, `conditions/depth/` 5000 (8-bit), `conditions/depth_raw/` 5000 (float16 npy) |
| `ade20k_val2k` | `/data/wookiekim/cgd/data/bench/ade20k_val2k/` | 644 MB | `images/` 2000 (cropped), `labels/` 2000, `conditions/seg/` 2000 |
| `coco_val5k` | `/data/wookiekim/cgd/data/bench/coco_val5k/` | 2.1 GB | `images/` 5000 (cropped), `conditions/bbox/` 5000 |
| `dreambench750` | `/data/wookiekim/cgd/data/bench/dreambench750/` | 22 MB | `refs/` 30 |
| `train/<set>/` | `/data/wookiekim/cgd/data/bench/train/<set>/` | pending | the CGD training triplets (`make_targets.py`), NOT built yet: waits for a GPU slot (~12 h on 4 H200s for ~174k images) |

Manifests for all of these are in this directory (`<set>/manifest.csv`, checksums in `BENCH.json`). The dev-200 set (`../dev200/`) is a subset of `multigen5k` (`dev200_idx`).

## 4. Models and scorers (HF cache unless noted)

| Role | HF id | Revision | Size | Notes |
|---|---|---|---|---|
| Generator | `black-forest-labs/FLUX.1-dev` | `3de623f` | 32 GB | gated; the container token has access |
| Controller: OminiControl | `Yuanshi/OminiControl` | `f482fe8` | 56 MB | `experimental/{canny,depth}.safetensors`, `omini/subject_512.safetensors` |
| Controller: EasyControl | `Xiaojiu-Z/EasyControl` | `0109ce2` | 2.3 GB | `models/{canny,depth,seg,subject,...}.safetensors` |
| Controller: FLUX ControlNet Union-Pro-2.0 | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` | `5d700aa` | 4.0 GB | canny, depth |
| Decoder: PiD distilled (baseline rows) | `nvidia/PiD` `checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/` | `1b9b087` | 2.6 GB | NOT in the HF cache: `/data/wookiekim/cgd/PiD/checkpoints/` |
| Decoder: PiD undistilled v1.5 (adapter training / training-free variant) | `nvidia/PiD` `checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux_undistilled/` | `1b9b087` | 2.7 GB | same location; `checkpoints/ae.safetensors` = FLUX VAE for PiD |
| PiD text encoder | `Efficient-Large-Model/gemma-2-2b-it` | `569d980` | 4.9 GB | |
| Seg scorer | `facebook/mask2former-swin-large-ade-semantic` | `aa25c92` | 1.7 GB | |
| Depth scorer and annotator | `Intel/dpt-large` | `bc15f29` | 2.6 GB | |
| Layout scorer | `IDEA-Research/grounding-dino-base` | `12bdfa3` | 1.8 GB | |
| Subject scorers | `facebook/dino-vits16`, `openai/clip-vit-large-patch14` | `abe3b35`, `32bd642` | 83 MB, 1.6 GB | |
| No-ref: DeQA-Score | `zhiyuanyou/DeQA-Score-Mix3` | `246e219` | 16 GB | safetensors only (`.bin` excluded); scorer code vendored in PiD |
| No-ref: Q-Align | `q-future/one-align` | `dcc603b` | 16 GB | via pyiqa `qalign` |
| No-ref: UniPercept IAA/IQA | `Thunderbolt215215/UniPercept` | `26cc5c8` | 15 GB | gated (auto-approve); accept the terms once |
| No-ref: VisualQuality-R1 | `TianheWu/VisualQuality-R1-7B` | `12852ce` | 16 GB | |
| No-ref: MUSIQ, NIQE, MANIQA; LPIPS, DISTS | pyiqa weights | | small | inside the container at `/root/.cache/torch/hub/pyiqa/` (auto-downloaded on first use elsewhere) |
| Realism | `clean-fid` (pip) | | | inception weights auto-download on first use |

All eight no-reference metrics of the paper's quality table are implemented in PiD's `pid/_src/evaluations/metrics/image_metrics.py` and load these ids; no separate scorer repos are needed.
