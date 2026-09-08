| Decoder | n | Canny F1 @512 (matched) ↑ | Canny F1 @2048 (native; VAE row via bilinear x4) ↑ | LPIPS ↓ | PSNR ↑ | SSIM ↑ | MUSIQ @512 (matched) ↑ | MUSIQ (native) ↑ | decode s/img ↓ | peak mem GB ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| OminiControl + VAE decode (512, native) | 200 | 0.3647 | 0.0263 (bilinear x4) | 0.5142 | 11.90 | 0.4394 | 70.40 | 70.40 | 0.033 | 34.5 |
| OminiControl + vanilla PiD (final latent, 2048) | 200 | 0.3624 | 0.2217 | 0.5092 | 11.81 | 0.4284 | 71.57 | 61.05 | 0.959 | 23.6 |
| OminiControl + vanilla PiD, early-terminated at 24/28 (2048) | 200 | 0.3005 | 0.2033 | 0.5265 | 11.76 | 0.3999 | 72.42 | 61.95 | 0.953 | 23.6 |
| OminiControl + vanilla PiD, early-terminated at 16/28 (2048) | 200 | 0.1916 | 0.1622 | 0.6002 | 11.67 | 0.3480 | 74.79 | 66.92 | 0.961 | 23.6 |

Generation (FLUX.1-dev + OminiControl canny LoRA, 28 steps @512, seed 0): 5.11 s/img, shared by every row.
