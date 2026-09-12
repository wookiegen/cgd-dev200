#!/usr/bin/env bash
# The NATIVE-ROUTE baseline (BENCHMARK v1.11 / v1.13; red text in the paper): each controller generating at 2048 DIRECTLY (FLUX at 4 MP) +
# the VAE decode, on subset500 of multigen5k, canny + depth. Scored like any 2048 output: 512 view vs c512, 2048 vs c512 (4 px), and for
# canny 2048 vs c2048 (1 px). Runs INSIDE the container on the GPUs in $GPUS (default "0 1 2 3"); ~2 to 4 min per image at 44 GB, so
# 3 controllers x 2 conditions x 500 images = 3000 generations ~ 30 GPU-h. Marker: _logs/done_native_route.txt
set -u
read -r -a G <<< "${GPUS:-0 1 2 3}"; NG=${#G[@]}
LOG=/data/wookiekim/cgd/data/_logs; B=/data/wookiekim/cgd/data/bench
cd /data/wookiekim/cgd/cgd-dev200/bench
echo "$(date '+%F %T') native route: generating at 2048 on GPUs ${G[*]}" > $LOG/native_route_status.txt
for ctrl in omini easycontrol fluxcn; do
  for c in canny depth; do
    for ((k=0; k<NG; k++)); do
      CUDA_VISIBLE_DEVICES=${G[$k]} python make_latents.py --controller $ctrl --condition $c --size 2048 --subset500 --out-tag ${ctrl}_2048 --capture-steps 24 --nshards $NG --shard $k 2>&1 \
        | grep -v -i -E "warning|pynvml|INFO|Time spent|PIL.Image" > $LOG/native_route_${ctrl}_${c}_$k.log &
    done
    wait
    echo "$(date '+%F %T') $ctrl/$c generated" >> $LOG/native_route_status.txt
  done
done
echo "$(date '+%F %T') generation done; scoring" >> $LOG/native_route_status.txt
score() { # gpu ctrl cond
  local g=$1 ctrl=$2 c=$3
  local O=$B/outputs/${ctrl}_2048/$c/vae@28   # separate statement (see run_teacher_block.sh)
  CUDA_VISIBLE_DEVICES=$g python harness.py --method vae --controller ${ctrl}_2048 --split multigen5k --condition $c --subset500 --gen-dir $O --res 512
  CUDA_VISIBLE_DEVICES=$g python harness.py --method vae --controller ${ctrl}_2048 --split multigen5k --condition $c --subset500 --gen-dir $O --res 2048 --metrics adherence,noref
  [ "$c" = canny ] && CUDA_VISIBLE_DEVICES=$g python harness.py --method vae --controller ${ctrl}_2048 --split multigen5k --condition $c --subset500 --gen-dir $O --res 2048 --metrics adherence --cond-res 2048
}
i=0
for ctrl in omini easycontrol fluxcn; do for c in canny depth; do
  ( score ${G[$((i % NG))]} $ctrl $c ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/native_route_score_${ctrl}_${c}.log &
  i=$((i+1))
done; done
wait
python assemble.py > /dev/null 2>&1; python make_baselines.py > /dev/null 2>&1
echo "$(date '+%F %T') NATIVE ROUTE DONE" > $LOG/done_native_route.txt
