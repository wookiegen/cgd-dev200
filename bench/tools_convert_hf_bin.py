"""Convert a cached HF model's single pytorch_model.bin to model.safetensors inside its snapshot (transformers refuses torch.load below
torch 2.6). Runs on CPU; the .bin is kept. Usage: python tools_convert_hf_bin.py facebook/dino-vits16"""
import os
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

repo = sys.argv[1]
snap = snapshot_download(repo, local_files_only=True)
src = os.path.join(snap, "pytorch_model.bin"); dst = os.path.join(snap, "model.safetensors")
if os.path.exists(dst):
    raise SystemExit(f"{dst} exists")
sd = torch.load(src, map_location="cpu", weights_only=True)
clean, seen = {}, {}
for k, v in sd.items():
    v = v.contiguous(); key = (v.data_ptr(), tuple(v.shape), v.dtype)
    if key in seen:
        v = v.clone()
    seen[key] = k; clean[k] = v
save_file(clean, dst, metadata={"format": "pt"})
print("wrote", dst, len(clean), "tensors")
