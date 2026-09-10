#!/usr/bin/env bash
# Re-run the standard-metric scoring of the OminiControl block with per-condition result files (the first pass let the canny and depth
# runs overwrite each other's <method>@<res>.json). GPU 0 = canny, GPU 1 = depth. Runs INSIDE the container.
set -u
LOG=/data/wookiekim/cgd/data/_logs
B=/data/wookiekim/cgd/data/bench
cd /data/wookiekim/cgd/cgd-dev200/bench
rm -f ../results/bench/multigen5k/vae@512.json ../results/bench/multigen5k/vae@512_per_image.csv ../results/bench/multigen5k/pid_k*@512.json ../results/bench/multigen5k/pid_k*@512_per_image.csv ../results/bench/multigen5k/pid_k*@2048.json ../results/bench/multigen5k/pid_k*@2048_per_image.csv
std() { # gpu condition
  local g=$1 c=$2 O=$B/outputs/omini/$2
  CUDA_VISIBLE_DEVICES=$g python harness.py --method vae --controller omini --split multigen5k --condition $c --gen-dir $O/vae@28 --res 512
  for K in 28 24 16; do
    CUDA_VISIBLE_DEVICES=$g python harness.py --method pid_k$K --controller omini --split multigen5k --condition $c --gen-dir $O/pid@$K --gen-dir-512 $O/pid@${K}_512 --res 512
    CUDA_VISIBLE_DEVICES=$g python harness.py --method pid_k$K --controller omini --split multigen5k --condition $c --gen-dir $O/pid@$K --res 2048 --metrics adherence,noref
  done
}
( std 0 canny ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/omini_rescore_canny.log &
( std 1 depth ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/omini_rescore_depth.log &
wait
python assemble.py > /dev/null 2>&1
echo "$(date '+%F %T') OMINI STD RESCORE DONE" > $LOG/done_omini_rescore.txt
