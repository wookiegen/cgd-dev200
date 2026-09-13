#!/usr/bin/env bash
# Item 1 (user, 2026-09-13; OPEN_QUESTIONS 13 fairness step): OminiControl SEGMENTATION adapter with the official recipe on ADE20K train
# (bench/omini_seg/train_seg.py + seg_512.yaml, one GPU), then the OminiControl seg block on ade20k_val2k (latents, VAE + PiD K = 28 / 24 / 16,
# scoring at 512 and 2048), then the conditioned-VAE archetype for seg (additive + modulated). Runs INSIDE the container.
# GPUS (default "0 1 2"): the first is the training GPU; all are used for generation / decoding / scoring afterwards.
# Marker: _logs/done_item1_omini_seg.txt. Fill = tab:main OminiControl seg cells, tab:recon seg group (switch from EasyControl), 2x2 seg column,
# remove the red EasyControl-seg texts (Method + Experiments), Implementation Details TODO (rank 4, 10k steps x 2 accum, batch 1).
set -u
read -r -a G <<< "${GPUS:-0 1 2}"; NG=${#G[@]}
L=/data/wookiekim/cgd/data/_logs; B=/data/wookiekim/cgd/data/bench; REPO=/data/wookiekim/cgd/cgd-dev200; OM=/data/wookiekim/cgd/OminiControl
st() { echo "$(date '+%F %T') $*" >> $L/item1_status.txt; }
echo "$(date '+%F %T') item 1 (OminiControl seg adapter): waiting for the ADE20K training crops" > $L/item1_status.txt
until [ -f $L/done_dryrun_crops_ade20k_train.txt ]; do sleep 60; done
st "crops ready: $(cat $L/done_dryrun_crops_ade20k_train.txt); training on GPU ${G[0]}"
cd $OM && CUDA_VISIBLE_DEVICES=${G[0]} OMINI_CONFIG=$REPO/bench/omini_seg/seg_512.yaml PYTHONPATH=$REPO/bench/omini_seg WANDB_MODE=disabled TOKENIZERS_PARALLELISM=true \
  accelerate launch --main_process_port 41359 -m train_seg 2>&1 | grep -v -i -E "warning|pynvml|deprecat" > $L/omini_seg_train.log
RUN=$(ls -td /data/wookiekim/cgd/data/omini_seg/runs/*/ | head -1); CKDIR=$(ls -d $RUN/ckpt/* | sort -t/ -k9 -n | tail -1)
LORA=$CKDIR/default.safetensors; st "training done; adapter $LORA"
[ -f "$LORA" ] || { st "ERROR: no adapter found"; exit 1; }
echo "$LORA" > /data/wookiekim/cgd/data/omini_seg/SEG_LORA_PATH.txt
# ---- OminiControl seg latents on ade20k_val2k (all GPUs)
cd $REPO/bench
for ((k=0; k<NG; k++)); do
  CUDA_VISIBLE_DEVICES=${G[$k]} python make_latents.py --controller omini --condition seg --split ade20k_val2k --omini-lora $LORA --nshards $NG --shard $k 2>&1 \
    | grep -v -i -E "warning|pynvml|INFO|Time spent|it/s" > $L/omini_seg_latents_$k.log &
done
wait
st "latents done ($(ls $B/latents/omini/seg | wc -l)); decoding + scoring the block"
GPUS="${G[*]}" bash run/run_block.sh omini "seg"
st "block scored; conditioned-VAE archetype for seg (additive on ${G[0]}, modulated on ${G[1 % NG]})"
( CUDA_VISIBLE_DEVICES=${G[0]} EXTRA="--freeze-decoder --lr-branch 1e-4 --set ade20k_train" bash run/run_cond_vae.sh seg ) &
( CUDA_VISIBLE_DEVICES=${G[1 % NG]} VARIANT=mod EXTRA="--freeze-decoder --lr-branch 1e-4 --mode mod --set ade20k_train" bash run/run_cond_vae.sh seg ) &
wait
python assemble.py > /dev/null 2>&1; python make_baselines.py > /dev/null 2>&1; chmod -R a+rwX $REPO/results $REPO/BASELINES.md /data/wookiekim/cgd/data/omini_seg 2>/dev/null
st "ITEM 1 DONE"
echo "$(date '+%F %T') ITEM 1 DONE (OminiControl seg adapter $LORA; block + archetype scored)" > $L/done_item1_omini_seg.txt
