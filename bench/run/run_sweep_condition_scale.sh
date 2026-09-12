#!/usr/bin/env bash
# Condition-scale sweep for fig:ceiling (canny dev-200, OminiControl): generate + PiD-decode on the GPUs in $GPUS (default "0 1"),
# one shard per GPU over the scales, then assemble on the CPU. Runs INSIDE the container, detached. Marker: _logs/done_sweep_condition_scale.txt
set -u
read -r -a G <<< "${GPUS:-0 1}"; NG=${#G[@]}
SCALES=${SCALES:-0,0.25,0.5,0.75,1,1.5,2,4}
LOG=/data/wookiekim/cgd/data/_logs
cd /data/wookiekim/cgd/cgd-dev200/bench
echo "$(date '+%F %T') sweep condition_scale: scales $SCALES on GPUs ${G[*]}" > $LOG/sweep_condition_scale_status.txt
for ((k=0; k<NG; k++)); do
  CUDA_VISIBLE_DEVICES=${G[$k]} python sweep_condition_scale.py --scales "$SCALES" --shard $k --nshards $NG 2>&1 \
    | grep -v -i -E "warning|pynvml|INFO|Time spent" > $LOG/sweep_condition_scale_shard$k.log &
done
wait
echo "$(date '+%F %T') generation done; assembling" >> $LOG/sweep_condition_scale_status.txt
python sweep_condition_scale.py --scales "$SCALES" --assemble > $LOG/sweep_condition_scale_assemble.log 2>&1
echo "$(date '+%F %T') SWEEP DONE" > $LOG/done_sweep_condition_scale.txt
