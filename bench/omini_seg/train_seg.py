"""Train an OminiControl segmentation adapter (LoRA) on ADE20K train with OminiControl's official trainer (BENCHMARK v1.8.1 / paper
Implementation Details: "for OminiControl we train a low-rank adapter with its official recipe on ADE20K train, with the mask rendered as a
class-color-coded image"). Removes the red EasyControl-seg note once the block is scored.

Data: bench/train/ade20k_train/manifest.csv (split == train, blocked != 1) -> $CGD_BENCH_ROOT/train/ade20k_train/images512/<sid>.png (target)
and conditions/seg/<sid>.png (ADE20K palette render of the v1.7 crop; the same renderer as ade20k_val2k/conditions/seg). Caption = manifest.
Recipe = train/config/spatial_alignment.yaml of OminiControl (LoRA r 4, Prodigy, batch 1, gradient checkpointing, drop text / image 0.1,
position_delta (0, 0) = spatially aligned), condition_type "seg"; steps / accumulation in bench/omini_seg/seg_512.yaml.
Usage (inside the container, from the OminiControl root so that `omini` imports):
  cd /data/wookiekim/cgd/OminiControl && OMINI_CONFIG=/data/wookiekim/cgd/cgd-dev200/bench/omini_seg/seg_512.yaml \
  PYTHONPATH=/data/wookiekim/cgd/cgd-dev200/bench/omini_seg CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 41359 -m train_seg
Saves <save_path>/<run>/ckpt/<step>/default.safetensors (load with pipe.load_lora_weights(dir, weight_name="default.safetensors", adapter_name="seg")).
"""
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset

PngImagePlugin.MAX_TEXT_CHUNK = 256 * 1024 * 1024

from omini.pipeline.flux_omini import Condition, generate  # noqa: E402
from omini.train_flux.trainer import OminiModel, get_config, train  # noqa: E402

BENCH_ROOT = Path(os.environ.get("CGD_BENCH_ROOT", "/data/wookiekim/cgd/data/bench"))
REPO_BENCH = Path(os.environ.get("CGD_REPO_BENCH", "/data/wookiekim/cgd/cgd-dev200/bench"))


class ADE20KSegDataset(Dataset):
    def __init__(self, set_name="ade20k_train", condition_size=(512, 512), target_size=(512, 512), drop_text_prob=0.1, drop_image_prob=0.1, limit=0):
        T_ = BENCH_ROOT / "train" / set_name
        rows = [r for r in csv.DictReader(open(REPO_BENCH / "train" / set_name / "manifest.csv")) if r["blocked"] != "1" and r["split"] == "train"]
        self.items = [(T_ / "images512" / f"{r['sample_id']}.png", T_ / "conditions" / "seg" / f"{r['sample_id']}.png", r["caption"]) for r in rows]
        self.items = [it for it in self.items if it[0].exists() and it[1].exists()][: limit or None]
        self.condition_size, self.target_size = tuple(condition_size), tuple(target_size)
        self.drop_text_prob, self.drop_image_prob = drop_text_prob, drop_image_prob
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_p, cond_p, caption = self.items[idx]
        image = Image.open(img_p).convert("RGB").resize(self.target_size)
        condition_img = Image.open(cond_p).convert("RGB").resize(self.condition_size, Image.NEAREST)
        description = "" if random.random() < self.drop_text_prob else caption
        if random.random() < self.drop_image_prob:
            condition_img = Image.new("RGB", self.condition_size, (0, 0, 0))
        return {"image": self.to_tensor(image), "condition_0": self.to_tensor(condition_img), "condition_type_0": "seg",
                "position_delta_0": np.array([0, 0]), "description": description}


@torch.no_grad()
def test_function(model, save_path, file_name):
    """One fixed validation sample per interval: ade20k_val2k 00000 (its palette mask + caption)."""
    os.makedirs(save_path, exist_ok=True)
    adapter = model.adapter_names[2]
    rows = {r["sample_id"]: r for r in csv.DictReader(open(REPO_BENCH / "ade20k_val2k" / "manifest.csv"))}
    for sid in ["00000", "00001"]:
        cond = Image.open(BENCH_ROOT / "ade20k_val2k" / "conditions" / "seg" / f"{sid}.png").convert("RGB").resize((512, 512), Image.NEAREST)
        cap = rows[sid]["caption"] if sid in rows else "a photo"
        g = torch.Generator(device=model.device).manual_seed(0)
        res = generate(model.flux_pipe, prompt=cap, conditions=[Condition(cond, adapter, np.array([0, 0]))], height=512, width=512,
                       num_inference_steps=20, guidance_scale=3.5, generator=g)
        canvas = Image.new("RGB", (1024, 512)); canvas.paste(cond, (0, 0)); canvas.paste(res.images[0], (512, 0))
        canvas.save(os.path.join(save_path, f"{file_name}_seg_{sid}.jpg"), quality=90)


def main():
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    ds_cfg = training_config["dataset"]
    dataset = ADE20KSegDataset(set_name=ds_cfg.get("set", "ade20k_train"), condition_size=ds_cfg["condition_size"], target_size=ds_cfg["target_size"],
                               drop_text_prob=ds_cfg["drop_text_prob"], drop_image_prob=ds_cfg["drop_image_prob"], limit=int(os.environ.get("SEG_LIMIT", "0")))
    print(f"ADE20K seg dataset: {len(dataset)} training pairs", flush=True)
    trainable_model = OminiModel(flux_pipe_id=config["flux_path"], lora_config=training_config["lora_config"], device="cuda",
                                 dtype=getattr(torch, config["dtype"]), optimizer_config=training_config["optimizer"],
                                 model_config=config.get("model", {}), gradient_checkpointing=training_config.get("gradient_checkpointing", False))
    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()
