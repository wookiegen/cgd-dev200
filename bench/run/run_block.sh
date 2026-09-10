#!/usr/bin/env bash
# Decode + score one controller's block (generalization of run_omini_block.sh). Usage: run_block.sh <controller> "<cond> <cond> ..."
# Conditions canny/depth live on multigen5k, seg on ade20k_val2k. Requires latents/<controller>/<cond>/ complete. Runs INSIDE the container.
set -u
CTRL=$1; CONDS=$2
LOG=/data/wookiekim/cgd/data/_logs
B=/data/wookiekim/cgd/data/bench
cd /data/wookiekim/cgd/cgd-dev200/bench
echo "$(date '+%F %T') $CTRL block: decoding $CONDS" > $LOG/${CTRL}_block_status.txt

# ---- decode: each condition sharded over 2 GPUs (canny -> 0,1; depth -> 2,3; seg -> 0,1 afterwards)
dec() { CUDA_VISIBLE_DEVICES=$1 python decode_pid.py --controller $CTRL --condition $2 --ks 28,24,16 --nshards 2 --shard $3 2>&1 | grep -v -i -E "warning|pynvml|INFO|Time spent" > $LOG/${CTRL}_block_decode_$2_$3.log; }
gpus=(0 1 2 3); i=0
for c in $CONDS; do
  if [ "$c" != "seg" ]; then dec ${gpus[$i]} $c 0 & dec ${gpus[$((i+1))]} $c 1 & i=$(( (i+2) % 4 )); fi
done
wait
for c in $CONDS; do
  if [ "$c" = "seg" ]; then dec 0 seg 0 & dec 1 seg 1 & wait; fi
done
echo "$(date '+%F %T') decodes complete; scoring" >> $LOG/${CTRL}_block_status.txt

# ---- score
std() { # gpu condition split
  local g=$1 c=$2 s=$3 O=$B/outputs/$CTRL/$2
  local extra=""; [ "$c" = "seg" ] && extra="--metrics adherence,recon,fid"
  CUDA_VISIBLE_DEVICES=$g python harness.py --method vae --controller $CTRL --split $s --condition $c --gen-dir $O/vae@28 --res 512 $extra
  for K in 28 24 16; do
    CUDA_VISIBLE_DEVICES=$g python harness.py --method pid_k$K --controller $CTRL --split $s --condition $c --gen-dir $O/pid@$K --gen-dir-512 $O/pid@${K}_512 --res 512 $extra
    if [ "$c" = "seg" ]; then
      CUDA_VISIBLE_DEVICES=$g python harness.py --method pid_k$K --controller $CTRL --split $s --condition $c --gen-dir $O/pid@$K --res 2048 --metrics adherence
    else
      CUDA_VISIBLE_DEVICES=$g python harness.py --method pid_k$K --controller $CTRL --split $s --condition $c --gen-dir $O/pid@$K --res 2048 --metrics adherence,noref
    fi
  done
}
vlm() { # gpu (canny rows only, tab:noref)
  local g=$1 O=$B/outputs/$CTRL/canny
  CUDA_VISIBLE_DEVICES=$g python harness.py --method vae --controller $CTRL --split multigen5k --condition canny --gen-dir $O/vae@28 --res 512 --metrics vlm
  for K in 28 24 16; do
    CUDA_VISIBLE_DEVICES=$g python harness.py --method pid_k$K --controller $CTRL --split multigen5k --condition canny --gen-dir $O/pid@$K --gen-dir-512 $O/pid@${K}_512 --res 512 --metrics vlm
    CUDA_VISIBLE_DEVICES=$g python harness.py --method pid_k$K --controller $CTRL --split multigen5k --condition canny --gen-dir $O/pid@$K --res 2048 --metrics vlm
  done
}
g=0
for c in $CONDS; do
  s=multigen5k; [ "$c" = "seg" ] && s=ade20k_val2k
  ( std $g $c $s ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/${CTRL}_block_score_$c.log &
  g=$((g+1))
done
if echo "$CONDS" | grep -q canny; then ( vlm $g ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/${CTRL}_block_score_vlm.log & fi
wait
python assemble.py > /dev/null 2>&1
echo "$(date '+%F %T') $CTRL BLOCK DONE" > $LOG/done_${CTRL}_block.txt
