#!/usr/bin/env bash
# Queue behind the OminiControl block: once its decodes are done (GPUs then only score), generate the EasyControl and FLUX ControlNet
# latent caches (one condition per GPU), then EasyControl seg on ADE20K, then decode + score each block. Runs INSIDE the container.
set -u
LOG=/data/wookiekim/cgd/data/_logs
cd /data/wookiekim/cgd/cgd-dev200/bench
until grep -q "decodes complete" $LOG/omini_block_status.txt 2>/dev/null; do sleep 300; done
echo "$(date '+%F %T') omini decodes done; generating easycontrol + fluxcn latents" > $LOG/queue_status.txt

gen() { # gpu controller condition split
  CUDA_VISIBLE_DEVICES=$1 python make_latents.py --controller $2 --condition $3 --split $4 2>&1 | grep -v -i -E "warning|pynvml|Loading pipeline|Loading checkpoint|deprecat|PIL.Image" > $LOG/latents_$2_$3.log
  echo "$(date '+%F %T') done" > $LOG/done_latents_$2_$3.txt
}
( gen 0 easycontrol canny multigen5k; gen 0 easycontrol seg ade20k_val2k ) &
( gen 1 easycontrol depth multigen5k ) &
( gen 2 fluxcn canny multigen5k ) &
( gen 3 fluxcn depth multigen5k ) &
wait
echo "$(date '+%F %T') all easycontrol + fluxcn latents done; blocks" >> $LOG/queue_status.txt
bash /data/wookiekim/cgd/tmp/run_block.sh easycontrol "canny depth seg"
bash /data/wookiekim/cgd/tmp/run_block.sh fluxcn "canny depth"
echo "$(date '+%F %T') QUEUE DONE" > $LOG/done_queue.txt
