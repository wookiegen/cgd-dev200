#!/usr/bin/env bash
# Score the reference rows once make_reference_rows.py has finished (done_ref_all.txt). Runs INSIDE the container.
# GPU 1: multigen5k (VAE round trip @512; PiD round trip @512 view and @2048; with no-ref incl. Q-Align, recon, FID)
# GPU 2: ade20k_val2k and coco_val5k (adherence + recon + FID at 512; PiD round trip also at 2048 adherence)
set -u
LOG=/data/wookiekim/cgd/data/_logs
B=/data/wookiekim/cgd/data/bench/outputs/ref
cd /data/wookiekim/cgd/cgd-dev200/bench
(
  CUDA_VISIBLE_DEVICES=1 python harness.py --method vae_roundtrip --split multigen5k --gen-dir $B/vae_roundtrip/multigen5k --res 512
  CUDA_VISIBLE_DEVICES=1 python harness.py --method pid_roundtrip --split multigen5k --gen-dir $B/pid_roundtrip/multigen5k --gen-dir-512 $B/pid_roundtrip_512/multigen5k --res 512
  CUDA_VISIBLE_DEVICES=1 python harness.py --method pid_roundtrip --split multigen5k --gen-dir $B/pid_roundtrip/multigen5k --res 2048 --metrics adherence,noref
  echo "$(date '+%F %T') gpu1 scoring done" > $LOG/done_score_ref_gpu1.txt
) > $LOG/score_ref_gpu1.log 2>&1 &
(
  for s in ade20k_val2k coco_val5k; do
    CUDA_VISIBLE_DEVICES=2 python harness.py --method vae_roundtrip --split $s --gen-dir $B/vae_roundtrip/$s --res 512 --metrics adherence,recon,fid
    CUDA_VISIBLE_DEVICES=2 python harness.py --method pid_roundtrip --split $s --gen-dir $B/pid_roundtrip/$s --gen-dir-512 $B/pid_roundtrip_512/$s --res 512 --metrics adherence,recon,fid
    CUDA_VISIBLE_DEVICES=2 python harness.py --method pid_roundtrip --split $s --gen-dir $B/pid_roundtrip/$s --res 2048 --metrics adherence
  done
  echo "$(date '+%F %T') gpu2 scoring done" > $LOG/done_score_ref_gpu2.txt
) > $LOG/score_ref_gpu2.log 2>&1 &
wait
echo "$(date '+%F %T') ALL REFERENCE SCORING DONE" > $LOG/done_score_ref_all.txt
