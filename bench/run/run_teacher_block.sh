#!/usr/bin/env bash
# BENCHMARK v1.11: the undistilled PiD TEACHER (25 steps, CFG 5) against the 4-step distilled student on subset500 (paired), OminiControl
# canny + depth at K = 28 / 24 / 16, plus the teacher round trip and the teacher latency row. Runs INSIDE the container on the GPUs in
# $GPUS (default "0 1 2 3"). ~25 s per 2048 decode -> ~3000 decodes + 500 round trips = ~25 GPU-h. Marker: _logs/done_teacher_block.txt
set -u
read -r -a G <<< "${GPUS:-0 1 2 3}"; NG=${#G[@]}
LOG=/data/wookiekim/cgd/data/_logs; B=/data/wookiekim/cgd/data/bench
cd /data/wookiekim/cgd/cgd-dev200/bench
echo "$(date '+%F %T') teacher block: PHASE 1 dev-200 (teacher decodes on ${G[0]}; OminiControl generating at 2048 on ${G[*]:1})" > $LOG/teacher_block_status.txt
# ---- PHASE 1: the colleagues' dev-200 reference table (user priority 2026-09-12): teacher rows + the native-route VAE row
REPO=/data/wookiekim/cgd/cgd-dev200
CUDA_VISIBLE_DEVICES=${G[0]} python $REPO/scripts/02b_decode_teacher.py 2>&1 | grep -v -i -E "warning|pynvml|it/s" > $LOG/dev200_teacher.log &
NR=$((NG-1)); for ((k=1; k<NG; k++)); do
  CUDA_VISIBLE_DEVICES=${G[$k]} python $REPO/scripts/01b_generate_omini_2048.py --shard $((k-1)) --nshards $NR 2>&1 | grep -v -i -E "warning|pynvml|it/s" > $LOG/dev200_gen2048_$((k-1)).log &
done
wait
echo "$(date '+%F %T') phase 1 decodes done; scoring dev-200" >> $LOG/teacher_block_status.txt
( cd $REPO && CUDA_VISIBLE_DEVICES=${G[0]} python scripts/03_score.py && python scripts/04_fill_readme.py && chmod -R a+rwX results README.md ) 2>&1 | grep -v -i -E "warning|pynvml" > $LOG/dev200_score.log
echo "$(date '+%F %T') DEV200 TABLE DONE" > $LOG/done_dev200_final.txt
echo "$(date '+%F %T') PHASE 2: subset500 teacher decodes on GPUs ${G[*]}" >> $LOG/teacher_block_status.txt
for c in canny depth; do
  for ((k=0; k<NG; k++)); do
    CUDA_VISIBLE_DEVICES=${G[$k]} python decode_pid.py --controller omini --condition $c --ks 28,24,16 --pid-ckpt-type teacher --subset500 --nshards $NG --shard $k 2>&1 \
      | grep -v -i -E "warning|pynvml|INFO|Time spent|it/s" > $LOG/teacher_decode_${c}_$k.log &
  done
  wait
done
echo "$(date '+%F %T') decodes done; teacher round trip" >> $LOG/teacher_block_status.txt
for ((k=0; k<NG; k++)); do
  CUDA_VISIBLE_DEVICES=${G[$k]} python make_reference_rows.py --split multigen5k --pid-ckpt-type teacher --subset500 --no-vae --nshards $NG --shard $k 2>&1 \
    | grep -v -i -E "warning|pynvml|INFO|Time spent|it/s" > $LOG/teacher_roundtrip_$k.log &
done
wait
echo "$(date '+%F %T') round trip done; scoring" >> $LOG/teacher_block_status.txt
score() { # gpu condition
  local g=$1 c=$2
  local O=$B/outputs/omini/$c   # separate statement: in one 'local' line $c would expand to the OUTER c (bug found 2026-09-12: canny rows scored the depth decodes)
  for K in 28 24 16; do
    for who in pidt pid; do                     # teacher and student on the SAME subset (paired table)
      local m=${who}_k$K
      CUDA_VISIBLE_DEVICES=$g python harness.py --method $m --controller omini --split multigen5k --condition $c --subset500 --gen-dir $O/${who}@$K --gen-dir-512 $O/${who}@${K}_512 --res 512
      CUDA_VISIBLE_DEVICES=$g python harness.py --method $m --controller omini --split multigen5k --condition $c --subset500 --gen-dir $O/${who}@$K --res 2048 --metrics adherence,noref
      [ "$c" = canny ] && CUDA_VISIBLE_DEVICES=$g python harness.py --method $m --controller omini --split multigen5k --condition $c --subset500 --gen-dir $O/${who}@$K --res 2048 --metrics adherence --cond-res 2048
    done
  done
}
( score ${G[0]} canny ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/teacher_score_canny.log &
( score ${G[1]} depth ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/teacher_score_depth.log &
( for who in pidt_roundtrip pid_roundtrip; do
    CUDA_VISIBLE_DEVICES=${G[2]} python harness.py --method $who --split multigen5k --condition canny,depth --subset500 --gen-dir $B/outputs/ref/$who/multigen5k --gen-dir-512 $B/outputs/ref/${who}_512/multigen5k --res 512
    CUDA_VISIBLE_DEVICES=${G[2]} python harness.py --method $who --split multigen5k --condition canny,depth --subset500 --gen-dir $B/outputs/ref/$who/multigen5k --res 2048 --metrics adherence,noref
  done ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/teacher_score_roundtrip.log &
( CUDA_VISIBLE_DEVICES=${G[3]} python measure_efficiency.py --only-pid ) 2>&1 | grep -v -i -E "warning|pynvml" > $LOG/teacher_efficiency.log &
wait
python assemble.py > /dev/null 2>&1; python make_baselines.py > /dev/null 2>&1
echo "$(date '+%F %T') TEACHER BLOCK DONE" > $LOG/done_teacher_block.txt
