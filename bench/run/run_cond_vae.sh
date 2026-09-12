#!/usr/bin/env bash
# Conditioned VAE decoder archetype, one condition end to end on ONE GPU: (depth: precompute the real-crop depth conditions) -> train ->
# decode the OminiControl multigen5k latents -> score at 512 (adherence, noref, recon, fid). Runs INSIDE the container, detached.
# Usage: CUDA_VISIBLE_DEVICES=g bash run_cond_vae.sh <canny|depth> [steps]      Marker: _logs/done_condvae_<cond>.txt
set -u
COND=$1; STEPS=${2:-6000}
LOG=/data/wookiekim/cgd/data/_logs; B=/data/wookiekim/cgd/data/bench
cd /data/wookiekim/cgd/cgd-dev200/bench
echo "$(date '+%F %T') condvae $COND: start (GPU $CUDA_VISIBLE_DEVICES)" > $LOG/condvae_${COND}_status.txt
if [ "$COND" = "depth" ]; then
  python precompute_real_depth.py --set multigen_train30 2>&1 | grep -v -i "warning|pynvml" > $LOG/condvae_depth_precompute.log
  echo "$(date '+%F %T') depth conditions done; training" >> $LOG/condvae_${COND}_status.txt
fi
python train_cond_vae.py --condition $COND --steps $STEPS --batch 8 --accum 1 ${EXTRA:-} 2>&1 | grep -v -i -E "warning|pynvml" > $LOG/condvae_${COND}_train.log   # EXTRA e.g. "--freeze-decoder --lr-branch 1e-4"
echo "$(date '+%F %T') training done; decoding" >> $LOG/condvae_${COND}_status.txt
python decode_cond_vae.py --condition $COND --controller omini 2>&1 | grep -v -i -E "warning|pynvml" > $LOG/condvae_${COND}_decode.log
echo "$(date '+%F %T') decode done; scoring" >> $LOG/condvae_${COND}_status.txt
python harness.py --method condvae --controller omini --split multigen5k --condition $COND --gen-dir $B/outputs/omini/$COND/condvae --res 512 2>&1 \
  | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/condvae_${COND}_score.log
python assemble.py > /dev/null 2>&1
echo "$(date '+%F %T') CONDVAE $COND DONE" > $LOG/done_condvae_${COND}.txt
