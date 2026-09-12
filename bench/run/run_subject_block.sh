#!/usr/bin/env bash
# DreamBench subject rows of tab:subject under one controller's released subject LoRA: generate latents (sharded over $GPUS), PiD-decode at
# K = 28 / 24 / 16, score the 512 view (DINO, CLIP-I, CLIP-T + no-ref). Runs INSIDE the container, detached.
# Usage: GPUS="2 3" bash run_subject_block.sh [omini|easycontrol]      Marker: _logs/done_subject_<ctrl>.txt
set -u
CTRL=${1:-omini}
read -r -a G <<< "${GPUS:-2 3}"; NG=${#G[@]}
LOG=/data/wookiekim/cgd/data/_logs
B=/data/wookiekim/cgd/data/bench
cd /data/wookiekim/cgd/cgd-dev200/bench
echo "$(date '+%F %T') subject block $CTRL: generating on GPUs ${G[*]}" > $LOG/subject_${CTRL}_status.txt
for ((k=0; k<NG; k++)); do
  CUDA_VISIBLE_DEVICES=${G[$k]} python make_subject_latents.py --controller $CTRL --shard $k --nshards $NG 2>&1 \
    | grep -v -i -E "warning|pynvml|INFO|Time spent" > $LOG/subject_${CTRL}_gen_$k.log &
done
wait
echo "$(date '+%F %T') latents done; decoding" >> $LOG/subject_${CTRL}_status.txt
for ((k=0; k<NG; k++)); do
  CUDA_VISIBLE_DEVICES=${G[$k]} python decode_pid.py --controller $CTRL --condition subject --ks 28,24,16 --nshards $NG --shard $k 2>&1 \
    | grep -v -i -E "warning|pynvml|INFO|Time spent" > $LOG/subject_${CTRL}_decode_$k.log &
done
wait
echo "$(date '+%F %T') decodes done; scoring" >> $LOG/subject_${CTRL}_status.txt
O=$B/outputs/$CTRL/subject
( CUDA_VISIBLE_DEVICES=${G[0]} python harness.py --method vae --controller $CTRL --split dreambench750 --gen-dir $O/vae@28 --res 512
  for K in 28 24 16; do
    CUDA_VISIBLE_DEVICES=${G[0]} python harness.py --method pid_k$K --controller $CTRL --split dreambench750 --gen-dir $O/pid@$K --gen-dir-512 $O/pid@${K}_512 --res 512
  done ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/subject_${CTRL}_score.log
python assemble.py > /dev/null 2>&1
echo "$(date '+%F %T') SUBJECT BLOCK $CTRL DONE" > $LOG/done_subject_${CTRL}.txt
