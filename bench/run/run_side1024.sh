#!/usr/bin/env bash
# OPEN_QUESTIONS 11, the 1024 side comparison: EasyControl and the FLUX ControlNet generate canny / depth at 1024 (their training resolution)
# on subset500, VAE decode, INTER_AREA to 512, scored against the SAME 512 conditions; the 512-generation VAE rows are scored on the same
# subset for the paired reading. Runs INSIDE the container on the GPUs in $GPUS (default "2 3"). Marker: _logs/done_side1024.txt
set -u
read -r -a G <<< "${GPUS:-2 3}"
LOG=/data/wookiekim/cgd/data/_logs; B=/data/wookiekim/cgd/data/bench
cd /data/wookiekim/cgd/cgd-dev200/bench
echo "$(date '+%F %T') side1024: generating on GPUs ${G[*]}" > $LOG/side1024_status.txt
for ctrl in easycontrol fluxcn; do
  CUDA_VISIBLE_DEVICES=${G[0]} python make_latents.py --controller $ctrl --condition canny --size 1024 --subset500 --out-tag ${ctrl}_1024 --capture-steps 24 2>&1 | grep -v -i -E "warning|pynvml|INFO|Time spent" > $LOG/side1024_${ctrl}_canny.log &
  CUDA_VISIBLE_DEVICES=${G[1]} python make_latents.py --controller $ctrl --condition depth --size 1024 --subset500 --out-tag ${ctrl}_1024 --capture-steps 24 2>&1 | grep -v -i -E "warning|pynvml|INFO|Time spent" > $LOG/side1024_${ctrl}_depth.log &
  wait
done
echo "$(date '+%F %T') generation done; scoring" >> $LOG/side1024_status.txt
( for ctrl in easycontrol fluxcn; do for c in canny depth; do
    CUDA_VISIBLE_DEVICES=${G[0]} python harness.py --method vae --controller ${ctrl}_1024 --split multigen5k --condition $c --subset500 --gen-dir $B/outputs/${ctrl}_1024/$c/vae@28 --res 512
    CUDA_VISIBLE_DEVICES=${G[0]} python harness.py --method vae --controller $ctrl --split multigen5k --condition $c --subset500 --gen-dir $B/outputs/$ctrl/$c/vae@28 --res 512
  done; done ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/side1024_score.log
python assemble.py > /dev/null 2>&1
echo "$(date '+%F %T') SIDE1024 DONE" > $LOG/done_side1024.txt
