# update.md: resolution and evaluation decisions (2026-09-12)

For colleagues and their coding agents. This file records the evaluation decisions made on 2026-09-12 so that a variant is built and
judged the way the paper will judge it. The authoritative spec is the paper repo's `docs/BENCHMARK_v1.md` (v1.9.1) and the experiments
draft's subsubsection "Scoring Protocol Across Resolutions"; this is the working summary.

**한 줄 요약.** 기본 설정에서 condition은 항상 512입니다. 평가는 512 matched view와 native 2048에서 따로 하며, edge F1은 "condition 픽셀 1개"
tolerance (512에서 1px, 2048에서 4px)를 씁니다. dev-200 게이트도 이 지표로 바뀌었습니다 (`git pull` 후 `scripts/03_score.py` 재실행). native
2048의 실질적 상한은 1.0이 아니라 PiD round trip의 0.80입니다. 여기에 더해 **native-condition track** (Section 7)이 추가되었습니다: 디코더에만
2048 condition을 주는 두 번째 설정으로, 학습 시 512 / 2048 두 형태를 반반 섞어 하나의 디코더가 둘 다 받을 수 있게 합니다. 논문의 헤드라인은
여전히 512-condition 프로토콜입니다.

## 1. What changed in the dev loop

- The gate is now the **tolerant canny F1** (one condition pixel: 1 px at the 512 matched view, 4 px at native 2048) at BOTH resolutions,
  against vanilla PiD at the same truncation point K. Strict pixel-exact F1 is still printed at 512 as an anchor; strict F1 at 2048 is not a
  gate (it measures the 4 px thick resampled reference, not content).
- Gate values (tolerant F1 to exceed, 512 / 2048): K=24 0.713 / 0.662; K=16 0.538 / 0.556; full latent 0.778 / 0.725. MUSIQ not lower,
  LPIPS not worse than vanilla PiD. Full table: README.md.
- Ids, cached latents, baseline outputs are unchanged. Pull commit d321661 or later and re-run `scripts/03_score.py`.

## 2. The scoring rule across resolutions

1. Every score is computed at a declared scoring resolution (512 matched view or native 2048) and compared only within it.
2. Matched view = `cv2.INTER_AREA` downsample of the native output to 512. Nothing is ever upsampled to be scored.
3. Edge F1 at resolution R: re-extract Canny (100, 200) from the output at R; reference = the 512 edge map resampled with NEAREST; a match
   counts within one condition pixel = R/512 output pixels. Thresholds are NOT rescaled with resolution.
4. Depth (DPT-Large), segmentation (Mask2Former), subject (DINO / CLIP) scorers resize internally, so their scores are resolution-invariant;
   they are reported at the matched view, native values only as a check.
5. No-reference quality (MUSIQ etc.) is compared within one resolution group only. FID / pFID / PSNR-SSIM-LPIPS use the 512 view (paired
   real images exist at 512 only).

Why the tolerance scales and the thresholds do not (control experiment on all 5000 real MultiGen images, upsampled bicubically and scored
against their own 512 edge map; `results/bench/metric_checks/upsampled_real_multigen5k.json`):

| scoring resolution | tolerant F1 | strict F1 | detected edge density |
|---|---|---|---|
| 512 (original) | 1.000 | 1.000 | 9.9% |
| 1024 (bicubic x2) | 0.859 | 0.469 | 5.5% |
| 2048 (bicubic x4) | 0.547 | 0.153 | 1.4% |
| 2048 then INTER_AREA back to 512 | 0.961 | 0.915 | 9.3% |

Reading: strict F1 at 2048 mostly measures edge thickness (0.15 for a perfect image); the tolerant score of a merely upsampled image is
0.55 because fixed Canny thresholds do not fire on interpolated (soft) edges. So the native column requires edges that are SHARP at native
scale and lie inside the condition's pixel blocks. A decoder that blurs upward will not score; that is intended.

**Why the tolerant F1 of a blind generative decoder drops from the 512 view to native 2048 (dev-200 decomposition, 2026-09-12):**

| row | precision @512 | recall @512 | precision @2048 | recall @2048 | F1 @512 -> @2048 |
|---|---|---|---|---|---|
| PiD round trip | 0.898 | 0.936 | 0.745 | 0.891 | 0.916 -> 0.793 |
| OminiControl + PiD 28/28 | 0.829 | 0.758 | 0.655 | 0.875 | 0.778 -> 0.725 |
| OminiControl + PiD K=24 | 0.691 | 0.763 | 0.571 | 0.864 | 0.713 -> 0.662 |

Recall does not drop (it even rises: the 2048 output is dense in edges, so most condition blocks have an edge nearby). The whole drop is
PRECISION: at native resolution 25 to 43 percent of the predicted edge pixels lie farther than one condition pixel from any condition edge,
i.e. detail the condition never asked for (texture, invented structure). The 512 view hides this because the INTER_AREA downsample averages
fine texture away. So the native column measures exactly what the paper claims a blind decoder does: it invents detail the condition did not
specify. A CGD variant should raise PRECISION at 2048 without losing recall. The harness now records `precision`, `recall`, and
`edge_density` per image and in the JSON (`bench/scorers.py`), so you can see which one you moved.

## 3. Native-resolution ceiling

The practical ceiling of the native 2048 column is **0.80**, the tolerant F1 of the vanilla PiD round trip (clean latent of the real image,
decoded by PiD, scored at 2048). Thin edges re-extracted at 2048 never align perfectly with the 4 px blocks of a condition defined at 512,
so no method reaches 1.0 there. Read native scores against 0.80, not 1.0. Vanilla PiD at K=24 under OminiControl scores 0.67; the gap to
0.80 is the headroom a condition-aware decoder can recover.

## 4. Indirect 2048 values for 512-native rows (double dagger)

Rows that only produce 512 (Real image, VAE round trip, each controller's VAE decode) get an indirect 2048 value: the tolerant F1 of their
BICUBIC x4 upsample, marked with a double dagger in the paper and "bicubic x4 ref." in the dev-200 table. It is the interpolation route to
2048, not a native output. Values on multigen5k: Real 0.547, VAE round trip 0.529, VAE decode under OminiControl 0.276 / EasyControl 0.439 /
ControlNet 0.305 (vs vanilla PiD K=24 native 0.669 / 0.766 / 0.649). Tool: `bench/tools_upsampled_ref.py`.

## 5. Conditions for CGD training and inference

- **The condition is always the 512 map** the generator received. The decoder gets the same map (pillar "condition twice"). If a variant
  injects at the pixel stream, resize the 512 map to the working resolution inside the decoder: NEAREST for edges / masks, BICUBIC for depth.
  A 4 px thick edge at 2048 is the honest representation: the condition says "an edge lies in this condition pixel", not where inside it.
- **Training triplets** (`bench/make_targets.py`, paper Method "Training the Conditioned Decoder"): input = the clean FLUX latent of the
  training crop re-noised to the truncation level; target = the 2048 vanilla-PiD decode of the clean latent; condition = extracted FROM THE
  TARGET at the 512 view (Canny / DPT on the INTER_AREA downsample of the target), so target and condition agree by construction (F1 = 1 at
  the matched view). Segmentation keeps the ground-truth mask (PiD's deviations are below mask scale).
- **Do NOT** extract a thin edge map from the 2048 target and feed it as the condition: it is more information than the generator received,
  and no such map exists at test time (train / test mismatch).
- **Allowed**: an auxiliary loss against the target's own 2048 edges. The target is ground truth; only the conditioning INPUT must stay at 512.
- The loss is on the 2048 pixels of the target, so nothing is lost in supervision; only the conditioning input is coarse, as it is at test time.
- Training data: MultiGen-20M train subset (canny, depth) and ADE20K train (seg); bounding-box layout was dropped from the paper on
  2026-09-12. Eval images are excluded by sha256 (`bench/blocklist/`); a 500-row `val` split per training set is for tuning.

## 6. Where things live

- Paper protocol text: paper repo `docs/EXPERIMENTS_draft.tex`, subsubsection `subsubsec:exp-resolution-protocol`; spec `docs/BENCHMARK_v1.md`
  (changelog v1.8, v1.9, v1.9.1); dev loop `docs/DEV_SETTING.md`.
- Scorers: `bench/scorers.py` (`CannyF1.score` returns `f1` tolerant and `f1_strict`), `bench/harness.py` (`--res 512|2048`, `--subset500`).
- Dev-200 scorer: `scripts/03_score.py` (`canny_f1_*` tolerant, `canny_f1s_*` strict).
- Paired significance: `bench/paired_ci.py` (paired bootstrap 95% CIs on per-image CSVs; on 5000 images the CIs are within +-0.003).
- Live state of all runs: `bench/STATUS.md`.

## 7. Native-condition setting (added 2026-09-12; FIRM framing decided the same evening)

**Framing (user decision, option 1).** 2048 is the REFERENCE resolution of the benchmark; the 512 view is derived from it. The matched
view exists only because the VAE decode produces nothing above 512. The cached latents stay as they are: the generator's condition is
Canny of the REAL 512 image, the native condition is Canny of the PiD round trip at 2048; the two agree at the 512 view at F1 0.92 (the
round trip's own 512 score), stated in the paper. No regeneration.

A second setting, supplementary in the paper: the DECODER receives the edge condition at the output resolution (2048), while the
generator keeps the 512 map. A latent-space controller cannot consume a 2048 map; a pixel-space decoder can. This is a strength of CGD we
keep, in addition to the default 512-condition protocol above (which remains the headline).

- **Eval conditions**: `multigen5k/conditions/canny2048/<sid>.png` = Canny(100, 200) of the vanilla-PiD round trip of the real image
  (`bench/build_native_conditions.py`). The controllers' input is unchanged, so every cached latent and decode is reused.
- **Scoring**: `harness.py ... --res 2048 --cond-res 2048` (tolerance one condition pixel = 1 px at 2048; file tag `.cond2048`). The PiD
  round trip is the reference and scores 1.0 by construction; dagger rows via `tools_upsampled_ref.py --cond-res 2048`. No FID against it.
- **Caveat (in the paper)**: the reference is synthesized by the blind decoder, so the track measures recovery of decoder-consistent native
  structure that the condition specifies, not agreement with a photograph. A real >= 2048 photo set would lift this; future work.
- **Training**: `make_targets.py` writes both `conditions/canny` (512 view of the target) and `conditions/canny2048` (target at 2048).
  Sample one of the two per example (p = 0.5) so ONE decoder accepts a condition at either scale; the 512 form is nearest-upsampled inside
  the decoder when the injection point is at 2048. Section 5 above stays valid for the default protocol: never feed a 2048-extracted map
  when the setting is "512 condition"; in the native-condition setting the 2048 map IS the condition, by definition.
- **Paper table**: `tab:native-cond` (supplementary): per controller {VAE decode (dagger), vanilla PiD K, CGD with the 512 condition, CGD
  with the 2048 condition}; the difference between the two CGD rows is the value of condition resolution.

**Baseline numbers against the native condition (tolerance 1 px at 2048; `results/bench/multigen5k/*.cond2048*.json`):**

| row | canny F1 @2048 vs native condition |
|---|---|
| PiD round trip (the reference) | 1.000 |
| real image, bicubic x4 (dagger) | 0.306 |
| VAE round trip, bicubic x4 (dagger) | 0.296 |
| OminiControl: VAE decode x4 (dagger) / PiD K=28 / K=24 / K=16 | 0.114 / 0.547 / 0.514 / 0.435 |
| EasyControl: VAE decode x4 (dagger) / PiD K=28 / K=24 / K=16 | 0.156 / 0.425 / 0.476 / 0.463 |
| FLUX ControlNet: VAE decode x4 (dagger) / PiD K=28 / K=24 / K=16 | 0.130 / 0.466 / 0.489 / 0.434 |

Reading: interpolation routes reach 0.11 to 0.31; the blind decoder recovers about half of the native structure from the truncated latent;
the reference is 1.0. This is the largest headroom of any setting for a condition-aware decoder. CGD rows: one with the 512 condition, one
with the 2048 condition (the difference = the value of condition resolution).

## 8. Queued / decided the same evening

- **PiD in two forms (BENCHMARK v1.11).** The main tables keep the released 4-step distilled STUDENT (`pid_k<K>`). The undistilled v1.5
  TEACHER (`pidt_k<K>`; `PiD_v1pt5_res2kto4k_sr4x_official_flux_undistilled`, PiD's documented teacher setting 25 steps + CFG 5, ~25 s per
  2048 decode) is the checkpoint CGD trains on; it is compared with the student on subset500 in a supplementary paired table
  (`tab:teacher`), plus a teacher round trip and a teacher latency row. Decoder: `decode_pid.py --pid-ckpt-type teacher --subset500`.
  For colleagues: build CGD on the TEACHER; the exact "CGD minus the condition" ablation is the `pidt` row.
- **Native-route baseline QUEUED** (controllers generating at 2048 directly, FLUX at 4 MP + VAE, subset500; red text in the paper).
- **Do not `pip install` into the container without `--no-deps`** (a plain install upgraded torch to 2.14 on 2026-09-12 and broke new
  processes until restored to 2.5.1+cu121).
