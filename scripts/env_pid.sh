#!/usr/bin/env bash
# One-shot: install PiD's python dependencies inside the container, excluding torch-family / cluster packages
# that the container already provides. PiD pins diffusers==0.37.1, transformers==4.57.1, numpy==1.26.4;
# running this CHANGES those versions in the shared container (see README "Caveats"). Use a venv if that matters.
set -euo pipefail
PID="${PID:-/data/wookiekim/cgd/PiD}"
python - <<EOF
import re, tomllib
deps = tomllib.load(open("$PID/pyproject.toml", "rb"))["project"]["dependencies"]
keep = [d for d in deps if not re.match(r"^(torch|torchvision|torchaudio|megatron|apex|flash.?attn|transformer.?engine|xformers|triton)", d, re.I)]
open("/tmp/pid_deps_f.txt", "w").write("\n".join(keep)); print("\n".join(keep))
EOF
pip install -r /tmp/pid_deps_f.txt
pip install pynvml   # imported by pid/_ext/imaginaire/utils/distributed.py
python -c "import pynvml, hydra, omegaconf, einops, loguru, fvcore, iopath, wandb; print('PiD deps OK')"
cd "$PID" && PYTHONPATH=. python -c "from pid._src.inference.checkpoint_registry import get_pid_checkpoint; print(get_pid_checkpoint('flux','2k'))"
