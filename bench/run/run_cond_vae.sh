#!/usr/bin/env bash
# Conditioned VAE decoder archetype, one condition end to end on ONE GPU: (depth: precompute the real-crop depth conditions) -> train ->
# decode the OminiControl multigen5k latents -> score at 512 (adherence, noref, recon, fid). Runs INSIDE the container, detached.
# Usage: CUDA_VISIBLE_DEVICES=g [VARIANT=<name>] [EXTRA="<train flags>"] bash run_cond_vae.sh <canny|depth> [steps]
#   VARIANT names the archetype variant (user, 2026-09-12: "two types"): default "" = the first run (additive injection, frozen decoder);
#     VARIANT=mod EXTRA="--freeze-decoder --lr-branch 1e-4 --mode mod"   SPADE-style modulation, frozen decoder      (the second TYPE)
#     VARIANT=ft  EXTRA="--lr 1e-5 --lr-branch 1e-4"                     additive injection, decoder finetuned at a safe lr (robustness)
#   Checkpoints cond_vae/<cond>[_<VARIANT>]/, decodes outputs/omini/<cond>/condvae[_<VARIANT>]/, method label condvae[_<VARIANT>].
#   Marker: _logs/done_condvae_<cond>[_<VARIANT>].txt
set -u
COND=$1; STEPS=${2:-6000}; V=${VARIANT:-}; SUF=${V:+_$V}
LOG=/data/wookiekim/cgd/data/_logs; B=/data/wookiekim/cgd/data/bench; CK=/data/wookiekim/cgd/data/cond_vae/${COND}${SUF}
cd /data/wookiekim/cgd/cgd-dev200/bench
echo "$(date '+%F %T') condvae $COND${SUF}: start (GPU $CUDA_VISIBLE_DEVICES; EXTRA=${EXTRA:-})" > $LOG/condvae_${COND}${SUF}_status.txt
if [ "$COND" = "depth" ]; then
  python precompute_real_depth.py --set multigen_train30 2>&1 | grep -v -i "warning|pynvml" > $LOG/condvae_depth${SUF}_precompute.log
  echo "$(date '+%F %T') depth conditions done; training" >> $LOG/condvae_${COND}${SUF}_status.txt
fi
python train_cond_vae.py --condition $COND --steps $STEPS --batch 8 --accum 1 --out $CK ${EXTRA:-} 2>&1 | grep -v -i -E "warning|pynvml" > $LOG/condvae_${COND}${SUF}_train.log
echo "$(date '+%F %T') training done; decoding" >> $LOG/condvae_${COND}${SUF}_status.txt
python decode_cond_vae.py --condition $COND --controller omini --ckpt $CK/best.safetensors --out-name condvae${SUF} 2>&1 | grep -v -i -E "warning|pynvml" > $LOG/condvae_${COND}${SUF}_decode.log
echo "$(date '+%F %T') decode done; scoring" >> $LOG/condvae_${COND}${SUF}_status.txt
python harness.py --method condvae${SUF} --controller omini --split multigen5k --condition $COND --gen-dir $B/outputs/omini/$COND/condvae${SUF} --res 512 2>&1 \
  | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/condvae_${COND}${SUF}_score.log
python assemble.py > /dev/null 2>&1
echo "$(date '+%F %T') CONDVAE $COND${SUF} DONE" > $LOG/done_condvae_${COND}${SUF}.txt
