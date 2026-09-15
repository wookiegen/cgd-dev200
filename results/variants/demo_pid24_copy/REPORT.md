# demo_pid24_copy: omini / canny / K=24 / subset500

| row | f1 @512 | @2048 | @2048 vs 2048 cond | MUSIQ @512 | MANIQA @512 | LPIPS | FID | pFID |
|---|---|---|---|---|---|---|---|---|
| VAE decode | 0.771 |  |  | 71.8 | 0.503 | 0.509 | 17.18 | 20.39 |
| vanilla PiD, K=24 | 0.719 | 0.669 | 0.514 | 73.0 | 0.650 | 0.521 | 18.09 | 19.33 |
| **demo_pid24_copy** | 0.716 | 0.669 | 0.510 | 73.1 | 0.647 | 0.521 | 79.53 | 83.96 |

Baseline rows above are the full-set records (n = 5000); the variant is n = 500, so its FID / pFID are NOT comparable with theirs (FID grows with fewer samples). The verdict below uses PAIRED means over the same 500 images.

## Verdict against vanilla PiD at the same K = 24 (paired on the same images)

- f1 @512 (higher better, strict inequality): 0.7160 vs 0.7160 (paired, n = 500) -> FAIL
- f1 @2048 (higher better): 0.6686 vs 0.6686 (paired, n = 500) -> FAIL
- MUSIQ @512 not lower (tol 0.5): 73.0683 vs 73.0683 (paired, n = 500) -> PASS
- LPIPS vs source not worse (tol 0.01): 0.5205 vs 0.5205 (paired, n = 500) -> PASS
- paired bootstrap, f1 @512: variant minus PiD = +0.0000, 95% CI [+0.0000, +0.0000], n = 500
- paired bootstrap, f1 @2048: variant minus PiD = +0.0000, 95% CI [+0.0000, +0.0000], n = 500
- paired bootstrap, f1 @2048 vs 2048 cond: variant minus PiD = +0.0000, 95% CI [+0.0000, +0.0000], n = 500

**GATE: FAIL** (all criteria above)

Baselines: BASELINES.md (regenerate with bench/make_baselines.py). Protocol: update.md.
