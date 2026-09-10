"""FLUX VAE encode/decode and vanilla PiD decode, shared by make_reference_rows.py (and reusable by make_targets.py / decoders).

Latent convention = diffusers' scaled space: z_scaled = (E(x) - shift) * scaling; PiD reads exactly this (its `extract_latent`), and the
VAE decode inverts it. PiD is loaded from $PID_ROOT with the release loader (cwd must be the PiD root, which load_pid handles).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

DEVICE = "cuda"


class FluxVAE:
    def __init__(self, dtype=torch.bfloat16):
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype).to(DEVICE).eval()
        self.sf, self.sh, self.dtype = self.vae.config.scaling_factor, self.vae.config.shift_factor, dtype

    @torch.no_grad()
    def encode(self, imgs: list[np.ndarray]) -> torch.Tensor:          # list of uint8 HxWx3 -> (B,16,H/8,W/8) fp16 scaled latents
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1) for im in imgs]).to(DEVICE, self.dtype) / 127.5 - 1.0
        z = self.vae.encode(x).latent_dist.mode()
        return ((z - self.sh) * self.sf).half()

    @torch.no_grad()
    def decode(self, lat: torch.Tensor) -> list[np.ndarray]:            # scaled latents -> list of uint8 HxWx3
        z = lat.to(DEVICE, self.dtype) / self.sf + self.sh
        x = self.vae.decode(z, return_dict=False)[0].float().clamp(-1, 1)
        return [((im.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).round().clip(0, 255).astype(np.uint8) for im in x]


class PiD:
    def __init__(self, pid_root: str | None = None, ckpt_type: str = "2k", load_ema_to_reg: bool = False):
        self.root = pid_root or os.environ.get("PID_ROOT", "/data/wookiekim/cgd/PiD")
        os.chdir(self.root); sys.path.insert(0, self.root)                    # PiD resolves checkpoints/ae.safetensors relative to its root
        from pid._src.inference.checkpoint_registry import get_pid_checkpoint  # noqa: E402
        from pid._src.utils.model_loader import load_model_from_checkpoint    # noqa: E402
        ck = get_pid_checkpoint("flux", ckpt_type)
        self.ckpt = ck
        self.model, _ = load_model_from_checkpoint(experiment_name=ck.experiment, checkpoint_path=ck.checkpoint_path,
                                                   config_file="pid/_src/configs/pid/config.py", enable_fsdp=False,
                                                   experiment_opts=[], strict=False, load_ema_to_reg=load_ema_to_reg)
        self.model.eval()
        self.cap_key = self.model.config.input_caption_key

    @torch.no_grad()
    def decode(self, lat: torch.Tensor, captions: list[str], sigma: float | list[float] = 0.0, steps: int = 4, seed: int = 0,
               cfg: float = 1.0, scale: int = 4) -> list[np.ndarray]:
        """lat: (B,16,h,w) scaled latents -> list of uint8 (h*8*scale) x (w*8*scale) x 3 images."""
        B = lat.shape[0]
        sig = torch.tensor([sigma] * B if isinstance(sigma, float) else list(sigma), device=DEVICE, dtype=torch.float32)
        batch = {self.cap_key: list(captions), "LQ_latent": lat.to(DEVICE, torch.bfloat16), "degrade_sigma": sig}
        hq = (lat.shape[-2] * 8 * scale, lat.shape[-1] * 8 * scale)
        s = self.model.generate_samples_from_batch(batch, cfg_scale=cfg, num_steps=steps, seed=seed, shift=None, image_size=hq)
        s = s.float().cpu().clamp(-1, 1)
        if s.ndim == 5:                                                        # (B,3,1,H,W)
            s = s[:, :, 0]
        return [((im.permute(1, 2, 0).numpy() + 1) / 2 * 255).round().clip(0, 255).astype(np.uint8) for im in s]
