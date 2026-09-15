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
