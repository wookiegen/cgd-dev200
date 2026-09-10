#!/usr/bin/env bash
# The OminiControl block of tab:main / tab:noref / tab:recon, fully automatic once the latent caches exist:
#   1. wait for done_latents_omini_all.txt
#   2. vanilla PiD decodes at K = 28 / 24 / 16 (canny shards on GPUs 0,1; depth shards on GPUs 2,3)
#   3. scoring: standard metrics for vae@28 and pid@K at 512 (and pid@K at 2048), both conditions; VLM metrics for the canny rows
# Runs INSIDE the container. Logs: _logs/omini_block_*.log ; marker: _logs/done_omini_block.txt
set -u
LOG=/data/wookiekim/cgd/data/_logs
B=/data/wookiekim/cgd/data/bench
cd /data/wookiekim/cgd/cgd-dev200/bench
until [ -f $LOG/done_latents_omini_all.txt ]; do sleep 120; done
echo "$(date '+%F %T') latents complete; decoding" > $LOG/omini_block_status.txt

dec() { CUDA_VISIBLE_DEVICES=$1 python decode_pid.py --controller omini --condition $2 --ks 28,24,16 --nshards 2 --shard $3 2>&1 | grep -v -i -E "warning|pynvml|INFO|Time spent" > $LOG/omini_block_decode_$2_$3.log; }
dec 0 canny 0 & dec 1 canny 1 & dec 2 depth 0 & dec 3 depth 1 &
wait
echo "$(date '+%F %T') decodes complete; scoring" >> $LOG/omini_block_status.txt

std() { # gpu condition
  local g=$1 c=$2 O=$B/outputs/omini/$2
  CUDA_VISIBLE_DEVICES=$g python harness.py --method vae --controller omini --split multigen5k --condition $c --gen-dir $O/vae@28 --res 512
  for K in 28 24 16; do
    CUDA_VISIBLE_DEVICES=$g python harness.py --method pid_k$K --controller omini --split multigen5k --condition $c --gen-dir $O/pid@$K --gen-dir-512 $O/pid@${K}_512 --res 512
    CUDA_VISIBLE_DEVICES=$g python harness.py --method pid_k$K --controller omini --split multigen5k --condition $c --gen-dir $O/pid@$K --res 2048 --metrics adherence,noref
  done
}
vlm() { # gpu (canny rows only, for tab:noref)
  local g=$1 O=$B/outputs/omini/canny
  CUDA_VISIBLE_DEVICES=$g python harness.py --method vae --controller omini --split multigen5k --condition canny --gen-dir $O/vae@28 --res 512 --metrics vlm
  for K in 28 24 16; do
    CUDA_VISIBLE_DEVICES=$g python harness.py --method pid_k$K --controller omini --split multigen5k --condition canny --gen-dir $O/pid@$K --gen-dir-512 $O/pid@${K}_512 --res 512 --metrics vlm
    CUDA_VISIBLE_DEVICES=$g python harness.py --method pid_k$K --controller omini --split multigen5k --condition canny --gen-dir $O/pid@$K --res 2048 --metrics vlm
  done
}
( std 0 canny ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/omini_block_score_canny.log &
( std 1 depth ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/omini_block_score_depth.log &
( vlm 2 ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/omini_block_score_vlm.log &
wait
python assemble.py > /dev/null 2>&1
echo "$(date '+%F %T') OMINI BLOCK DONE" > $LOG/done_omini_block.txt
