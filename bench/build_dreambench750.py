"""dreambench750: DreamBench (DreamBooth dataset), 30 subjects x 25 prompts = 750 (reference image, prompt) pairs.

Source: HF google/dreambooth (dataset/<subject>/*.jpg, dataset/prompts_and_classes.txt). Reference image = the first image (sorted) of each
subject. Prompt = the object or live-subject prompt list ('{0} {1}' -> class name, no unique token), 25 per subject. Live classes: cat, dog.
Writes: manifest.csv (repo; references are small and copied to refs/<subject>.jpg under the data root), VERSION.json.
"""
import re
import shutil
from pathlib import Path

from PIL import Image

from common import OUT_ROOT, RAW_ROOT, REPO_BENCH, env_pins, sha256_file, sid, write_json, write_manifest

SET = "dreambench750"
out = OUT_ROOT / SET
(out / "refs").mkdir(parents=True, exist_ok=True)
root = RAW_ROOT / "dreambench" / "dataset"
txt = open(root / "prompts_and_classes.txt").read()
classes = {k: v for k, v in re.findall(r"^([a-z0-9_]+),([a-z ]+)$", txt, flags=re.M) if k != "subject_name"}   # drop the CSV header line
assert len(classes) == 30, len(classes)
obj_part = txt[txt.index("Object Prompts"):txt.index("Live Subject Prompts")]
live_part = txt[txt.index("Live Subject Prompts"):]
obj_prompts = re.findall(r"'([^']*\{0\} \{1\}[^']*)'", obj_part)
live_prompts = re.findall(r"'([^']*\{0\} \{1\}[^']*)'", live_part)
assert len(obj_prompts) == 25 and len(live_prompts) == 25, (len(obj_prompts), len(live_prompts))
live_classes = {"cat", "dog"}
rows = []
i = 0
for subj in sorted(classes):
    cls = classes[subj]
    imgs = sorted(p for p in (root / subj).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    ref = imgs[0]
    shutil.copyfile(ref, out / "refs" / f"{subj}{ref.suffix.lower()}")
    w, h = Image.open(ref).size
    plist = live_prompts if cls in live_classes else obj_prompts
    for k, tpl in enumerate(plist):
        prompt = tpl.replace("{0} {1}", cls).replace("{0}", "").replace("{1}", cls).strip()
        prompt = re.sub(r"\s+", " ", prompt)
        rows.append({"sample_id": sid(i), "subject": subj, "class": cls, "live": int(cls in live_classes), "ref_image": ref.name,
                     "ref_sha256": sha256_file(ref), "ref_width": w, "ref_height": h, "n_subject_images": len(imgs), "prompt_idx": k,
                     "prompt_template": tpl, "prompt": prompt})
        i += 1
assert len(rows) == 750
cols = ["sample_id", "subject", "class", "live", "ref_image", "ref_sha256", "ref_width", "ref_height", "n_subject_images", "prompt_idx", "prompt_template", "prompt"]
write_manifest(REPO_BENCH / SET / "manifest.csv", rows, cols)
write_json(REPO_BENCH / SET / "VERSION.json", {
    "set": SET, "n": len(rows), "source": {"hf_dataset": "google/dreambooth", "github": "https://github.com/google/dreambooth", "prompts": "dataset/prompts_and_classes.txt"},
    "reference": "first image (sorted filename) of each subject; the other subject images are not used", "prompt": "'{0} {1}' -> the class name, no unique token; live classes (cat, dog) use the live list",
    "scorers": {"DINO": "facebook/dino-vits16", "CLIP-I/CLIP-T": "openai/clip-vit-large-patch14"}, "env": env_pins()})
print(f"{SET}: {len(rows)} rows, {len(classes)} subjects, live subjects {sum(1 for s, c in classes.items() if c in live_classes)}")
