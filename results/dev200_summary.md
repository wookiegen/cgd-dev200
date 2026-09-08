| Decoder | n | Canny F1 @512 (matched) | Canny F1 @2048 (native) | LPIPS | PSNR | SSIM | MUSIQ @512 (matched) | MUSIQ (native) | decode s/img | peak mem GB |
|---|---|---|---|---|---|---|---|---|---|---|
| OminiControl + VAE decode (512, native) | 200 | 0.3649 | n/a (512 native) | 0.5143 | 11.90 | 0.4394 | 70.41 | 70.41 | 0.033 | 34.5 |
| OminiControl + vanilla PiD (final latent, 2048) | 200 | 0.3625 | 0.2217 | 0.5092 | 11.81 | 0.4284 | 71.58 | 61.05 | 0.962 | 23.6 |
| OminiControl + vanilla PiD, early-terminated at 24/28 (2048) | 200 | 0.3003 | 0.2033 | 0.5266 | 11.76 | 0.3999 | 72.44 | 61.96 | 0.953 | 23.6 |

Generation (FLUX.1-dev + OminiControl canny LoRA, 28 steps @512, seed 0): 5.15 s/img, shared by every row.
