#!/usr/bin/env bash
# LSDIR validation of the c2048 column on REAL high-resolution photographs (BENCHMARK v1.13; red text in the paper, tab:lsdir).
# Split lsdir1k (bench/build_lsdir1k.py): 1000 LSDIR crops at 2048 (real reference), their 512 view, c512 = Canny(512), c2048 = Canny(real 2048).
# Rows: real 2048 (native, vs c512 at 4 px and vs c2048 at 1 px = 1.0), real 512 (+ bicubic x4 dagger), VAE round trip (+ dagger), PiD student
# round trip, OminiControl + VAE decode (+ dagger), + PiD student K = 28 / 24 / 16; every 2048 output in all three edge columns; FID / pFID at 512.
# Runs INSIDE the container on the GPUs in $GPUS (default "0 1 2 3"); ~2 GPU-h. Marker: _logs/done_lsdir_validation.txt
set -u
read -r -a G <<< "${GPUS:-0 1 2 3}"; NG=${#G[@]}
LOG=/data/wookiekim/cgd/data/_logs; B=/data/wookiekim/cgd/data/bench; S=lsdir1k
cd /data/wookiekim/cgd/cgd-dev200/bench
F="grep -v -i -E warning|pynvml|INFO|Time spent|it/s|PIL.Image"
st() { echo "$(date '+%F %T') $*" >> $LOG/lsdir_status.txt; }
echo "$(date '+%F %T') lsdir validation: start on GPUs ${G[*]}" > $LOG/lsdir_status.txt

# 0. the split itself (CPU + network); resumable; usually already built by the chain launcher
python build_lsdir1k.py --scan 2>&1 | grep -v -i -E "warning|pynvml" > $LOG/lsdir_scan.log
python build_lsdir1k.py --build 2>&1 | grep -v -i -E "warning|pynvml" > $LOG/lsdir_build.log
st "split built: $(ls $B/$S/images2048 | wc -l) images"
# 1. captions (one GPU) while the reference rows run on the others
CUDA_VISIBLE_DEVICES=${G[0]} python build_lsdir1k.py --captions 2>&1 | grep -v -i -E "warning|pynvml" > $LOG/lsdir_captions.log &
for ((k=1; k<NG; k++)); do
  CUDA_VISIBLE_DEVICES=${G[$k]} python make_reference_rows.py --split $S --nshards $((NG-1)) --shard $((k-1)) 2>&1 | grep -v -i -E "warning|pynvml|INFO|Time spent|it/s" > $LOG/lsdir_ref_$k.log &
done
wait
st "captions + reference rows done"
# 2. OminiControl canny latents (28 steps, 512) on all GPUs; outputs under outputs/omini_lsdir/canny (no collision with multigen5k ids)
for ((k=0; k<NG; k++)); do
  CUDA_VISIBLE_DEVICES=${G[$k]} python make_latents.py --controller omini --condition canny --split $S --out-tag omini_lsdir --nshards $NG --shard $k 2>&1 \
    | grep -v -i -E "warning|pynvml|INFO|Time spent|it/s" > $LOG/lsdir_latents_$k.log &
done
wait
st "latents done"
# 3. PiD student decodes at K = 28 / 24 / 16
for ((k=0; k<NG; k++)); do
  CUDA_VISIBLE_DEVICES=${G[$k]} python decode_pid.py --controller omini_lsdir --condition canny --ks 28,24,16 --nshards $NG --shard $k 2>&1 \
    | grep -v -i -E "warning|pynvml|INFO|Time spent|it/s" > $LOG/lsdir_decode_$k.log &
done
wait
st "decodes done; scoring"
# 4. scoring: three edge columns for every 2048 output; daggers for the 512-native rows; FID / pFID at 512 against lsdir1k/images
O=$B/outputs/omini_lsdir/canny; REF=$B/outputs/ref
three() { # gpu method controller gen2048 gen512
  local g=$1 m=$2 c=$3 d=$4 d5=$5
  CUDA_VISIBLE_DEVICES=$g python harness.py --method $m ${c:+--controller $c} --split $S --condition canny --gen-dir $d --gen-dir-512 $d5 --res 512
  CUDA_VISIBLE_DEVICES=$g python harness.py --method $m ${c:+--controller $c} --split $S --condition canny --gen-dir $d --res 2048 --metrics adherence,noref
  CUDA_VISIBLE_DEVICES=$g python harness.py --method $m ${c:+--controller $c} --split $S --condition canny --gen-dir $d --res 2048 --metrics adherence --cond-res 2048
}
dagger() { # method controller gen512
  python tools_upsampled_ref.py --split $S --method $1 ${2:+--controller $2} --gen-dir $3
  python tools_upsampled_ref.py --split $S --method $1 ${2:+--controller $2} --gen-dir $3 --cond-res 2048
}
( CUDA_VISIBLE_DEVICES=${G[0]} python harness.py --method real --split $S --condition canny --gen-dir $B/$S/images --res 512
  CUDA_VISIBLE_DEVICES=${G[0]} python harness.py --method real2048 --split $S --condition canny --gen-dir $B/$S/images2048 --res 2048 --metrics adherence,noref
  CUDA_VISIBLE_DEVICES=${G[0]} python harness.py --method real2048 --split $S --condition canny --gen-dir $B/$S/images2048 --res 2048 --metrics adherence --cond-res 2048
  CUDA_VISIBLE_DEVICES=${G[0]} python harness.py --method vae_roundtrip --split $S --condition canny --gen-dir $REF/vae_roundtrip/$S --res 512
  dagger real "" $B/$S/images; dagger vae_roundtrip "" $REF/vae_roundtrip/$S; dagger vae omini $O/vae@28
) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/lsdir_score_ref.log &
( three ${G[1 % NG]} pid_roundtrip "" $REF/pid_roundtrip/$S $REF/pid_roundtrip_512/$S
  CUDA_VISIBLE_DEVICES=${G[1 % NG]} python harness.py --method vae --controller omini --split $S --condition canny --gen-dir $O/vae@28 --res 512
) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/lsdir_score_rt.log &
( three ${G[2 % NG]} pid_k28 omini $O/pid@28 $O/pid@28_512; three ${G[2 % NG]} pid_k16 omini $O/pid@16 $O/pid@16_512 ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/lsdir_score_k28_16.log &
( three ${G[3 % NG]} pid_k24 omini $O/pid@24 $O/pid@24_512 ) 2>&1 | grep -v -i -E "warning|pynvml|Loading checkpoint" > $LOG/lsdir_score_k24.log &
wait
python assemble.py > /dev/null 2>&1; python make_baselines.py > /dev/null 2>&1
chmod -R a+rwX /data/wookiekim/cgd/cgd-dev200/results /data/wookiekim/cgd/cgd-dev200/bench/$S 2>/dev/null
st "LSDIR VALIDATION DONE"
echo "$(date '+%F %T') LSDIR VALIDATION DONE" > $LOG/done_lsdir_validation.txt
