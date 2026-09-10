#!/usr/bin/env bash
# Reference rows (VAE round trip + PiD round trip) for all three spatial eval sets, sharded over GPUs 0-2 (GPU 3 is kept for efficiency runs).
# Runs INSIDE the container. Logs: _logs/ref_gpu<g>.log ; marker _logs/done_ref_gpu<g>.txt
set -u
LOG=/data/wookiekim/cgd/data/_logs
cd /data/wookiekim/cgd/cgd-dev200/bench
for g in 0 1 2; do
  (
    for s in multigen5k ade20k_val2k coco_val5k; do
      echo "== $(date '+%F %T') $s shard $g"
      CUDA_VISIBLE_DEVICES=$g python make_reference_rows.py --split $s --shard $g --nshards 3 2>&1 | grep -v -i -E "warning|pynvml|Time spent|INFO"
    done
    echo "$(date '+%F %T') gpu$g done" > $LOG/done_ref_gpu$g.txt
  ) > $LOG/ref_gpu$g.log 2>&1 &
done
wait
echo "$(date '+%F %T') ALL REFERENCE ROWS DONE" > $LOG/done_ref_all.txt
