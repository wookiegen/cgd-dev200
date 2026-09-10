"""Convert q-future/one-align's pytorch_model-*.bin shards to safetensors inside the HF cache snapshot, so transformers (which refuses
torch.load below torch 2.6) can load Q-Align through pyiqa. Runs on CPU; the .bin files are kept."""
import glob, json, os
import torch
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

snap = snapshot_download("q-future/one-align", local_files_only=True)
print("snapshot:", snap)
bins = sorted(glob.glob(os.path.join(snap, "pytorch_model-*.bin")))
if not bins:
    raise SystemExit("no .bin shards found")
weight_map = {}
total = 0
for i, b in enumerate(bins, 1):
    sd = torch.load(b, map_location="cpu", weights_only=True)
    # safetensors needs contiguous, non-shared tensors
    clean = {}
    seen = {}
    for k, v in sd.items():
        v = v.contiguous()
        key = (v.data_ptr(), v.shape, v.dtype)
        if key in seen:
            v = v.clone()
        seen[key] = k
        clean[k] = v
        total += v.numel() * v.element_size()
    name = f"model-{i:05d}-of-{len(bins):05d}.safetensors"
    save_file(clean, os.path.join(snap, name), metadata={"format": "pt"})
    for k in clean:
        weight_map[k] = name
    print("wrote", name, len(clean), "tensors")
    del sd, clean
json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, open(os.path.join(snap, "model.safetensors.index.json"), "w"), indent=1)
print("index written; tensors:", len(weight_map), "bytes:", total)
