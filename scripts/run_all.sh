#!/usr/bin/env bash
# End-to-end dev-200 pipeline. Run inside the container (see README "Environment").
# Steps are idempotent: each skips work whose outputs already exist (pass --overwrite to redo).
set -euo pipefail
REPO="${REPO:-/data/wookiekim/cgd/cgd-dev200}"
GPU_GEN="${GPU_GEN:-0}"
GPU_DEC="${GPU_DEC:-1}"
cd "$REPO"

echo "== 00 select the frozen dev-200 (no GPU) =="
python scripts/00_select_dev200.py

echo "== 01 OminiControl(canny) latents + VAE-decode row (GPU $GPU_GEN) =="
CUDA_VISIBLE_DEVICES=$GPU_GEN python scripts/01_generate_latents_omini.py

echo "== 02 vanilla PiD decode: final + early-terminated (GPU $GPU_DEC) =="
CUDA_VISIBLE_DEVICES=$GPU_DEC python scripts/02_decode_pid.py

echo "== 03 score (GPU $GPU_DEC) =="
CUDA_VISIBLE_DEVICES=$GPU_DEC python scripts/03_score.py
echo "done -> results/dev200_summary.md"
