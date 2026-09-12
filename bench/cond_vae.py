"""Conditioned VAE decoder: the decoder-conditioning ARCHETYPE of BENCHMARK v1 (Group B; the condition-aware / deterministic cell of
tab:decoder-conditioning), re-implemented on the FLUX latent since the published methods (Asymmetric VQGAN, StableSR, RefDecoder) decode
other latent spaces.

Architecture: the FLUX VAE decoder (49.5M params) plus a condition branch. A small conv pyramid maps the 512x512 condition image (canny
edge map, depth map, or palette mask, 3 channels) to features at the decoder's four resolutions (64, 128, 256, 512); each is added to the
decoder's hidden state through a ZERO-initialised 1x1 conv, after the mid block and after the first three up blocks. At initialisation the
model is exactly the FLUX VAE decoder; training moves it toward using the condition. Deterministic, one forward pass, no noise, no prior.

Inputs follow diffusers: `decode(z)` takes the UNSCALED latent (the VAE's own space; for cached diffusers-scaled latents use
`lat / scaling_factor + shift_factor` first) and returns an image in [-1, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero(m: nn.Module) -> nn.Module:
    nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
    return m


class CondBranch(nn.Module):
    """Condition pyramid: 512 -> 512 / 256 / 128 / 64 feature maps (64 / 128 / 256 / 256 channels)."""

    def __init__(self, in_ch: int = 3, widths=(64, 128, 256, 256)):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.s512 = nn.Sequential(nn.Conv2d(in_ch, w0, 3, padding=1), nn.SiLU(), nn.Conv2d(w0, w0, 3, padding=1), nn.SiLU())
        self.s256 = nn.Sequential(nn.Conv2d(w0, w1, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(w1, w1, 3, padding=1), nn.SiLU())
        self.s128 = nn.Sequential(nn.Conv2d(w1, w2, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(w2, w2, 3, padding=1), nn.SiLU())
        self.s64 = nn.Sequential(nn.Conv2d(w2, w3, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(w3, w3, 3, padding=1), nn.SiLU())

    def forward(self, c: torch.Tensor) -> dict[int, torch.Tensor]:
        f512 = self.s512(c); f256 = self.s256(f512); f128 = self.s128(f256); f64 = self.s64(f128)
        return {512: f512, 256: f256, 128: f128, 64: f64}


class CondVAEDecoder(nn.Module):
    """Wraps a diffusers `Decoder` (FLUX VAE: block_out_channels [128, 256, 512, 512]; stream = 512ch@64 after mid, 512@128, 512@256,
    256@512, 128@512 after the four up blocks) and injects the condition after mid / up0 / up1 / up2."""

    def __init__(self, decoder: nn.Module, post_quant_conv: nn.Module | None = None, in_ch: int = 3, mode: str = "add"):
        """mode = "add": the condition features are ADDED to the hidden state (feature-fusion archetype, the Asymmetric VQGAN / StableSR family);
        mode = "mod": they MODULATE it, x * (1 + gamma) + beta with spatial gamma / beta (SPADE-style modulation archetype). Both zero-initialised."""
        super().__init__()
        assert mode in ("add", "mod"), mode
        self.dec = decoder; self.post_quant_conv = post_quant_conv; self.mode = mode
        self.branch = CondBranch(in_ch)
        w = self.branch.s512[0].out_channels, self.branch.s256[0].out_channels, self.branch.s128[0].out_channels, self.branch.s64[0].out_channels
        ups = decoder.up_blocks
        ch_after = [decoder.mid_block.resnets[-1].conv2.out_channels] + [b.resnets[-1].conv2.out_channels for b in ups[:3]]
        res_after = [64, 128, 256, 512]
        mk = lambda: nn.ModuleList([_zero(nn.Conv2d({64: w[3], 128: w[2], 256: w[1], 512: w[0]}[r], c, 1)) for r, c in zip(res_after, ch_after)])  # noqa: E731
        if mode == "add":
            self.inject = mk()
        else:
            self.inject_g = mk(); self.inject_b = mk()
        self.res_after = res_after

    def _inj(self, i: int, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        if self.mode == "add":
            return x + self.inject[i](f)
        return x * (1 + self.inject_g[i](f)) + self.inject_b[i](f)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """z: (B,16,h,w) unscaled latent; cond: (B,3,8h,8w) in [-1, 1]. Returns the image in [-1, 1]."""
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        feats = self.branch(cond)
        d = self.dec
        x = d.conv_in(z)
        x = d.mid_block(x, None)
        x = self._inj(0, x, feats[self.res_after[0]])
        for i, up in enumerate(d.up_blocks):
            x = up(x, None)
            if i < 3:
                x = self._inj(i + 1, x, feats[self.res_after[i + 1]])
        x = d.conv_norm_out(x); x = d.conv_act(x); x = d.conv_out(x)
        return x

    def cond_parameters(self):
        ps = list(self.branch.parameters())
        for n in ("inject", "inject_g", "inject_b"):
            if hasattr(self, n):
                ps += list(getattr(self, n).parameters())
        return ps

    def trainable_parameters(self, decoder: bool = True):
        ps = self.cond_parameters()
        if decoder:
            ps += list(self.dec.parameters())
        return ps

    def state_to_save(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("post_quant_conv")}


def cond_to_tensor(cond_rgb_u8, device) -> torch.Tensor:
    """uint8 HxWx3 numpy (or a batch list) -> (B,3,H,W) float in [-1, 1]."""
    import numpy as np
    arr = np.stack(cond_rgb_u8) if isinstance(cond_rgb_u8, (list, tuple)) else cond_rgb_u8[None]
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float().div(127.5).sub(1.0).to(device)


def load_cond_vae(vae, ckpt_path: str | None = None, device="cuda", dtype=torch.float32, mode: str | None = None) -> CondVAEDecoder:
    """Build the conditioned decoder around a diffusers AutoencoderKL (FLUX); optionally load a trained state dict (safetensors).
    The injection mode is read from the checkpoint keys (inject_g -> "mod") unless given."""
    sd = None
    if ckpt_path:
        from safetensors.torch import load_file
        sd = load_file(ckpt_path)
        mode = mode or ("mod" if any(k.startswith("inject_g") for k in sd) else "add")
    m = CondVAEDecoder(vae.decoder, getattr(vae, "post_quant_conv", None), mode=mode or "add")
    if sd is not None:
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("post_quant_conv") for k in missing), missing
    return m.to(device, dtype)
