# CGD 평가 프로토콜과 베이스라인 안내 (2026-09-12)

동료와 동료의 코딩 에이전트를 위한 요약입니다. 자세한 근거는 `update.md`, 숫자는 `BASELINES.md`, 실행 상태는 `bench/STATUS.md`에 있습니다.

## 1. 무엇을 이겨야 하나

`BASELINES.md`의 표 A가 기준입니다. 같은 latent (`latents/<controller>/<condition>/<sid>.pt`, x_0와 x_t@24, x_t@16 캐시)를 같은 K에서
디코딩해서, **같은 K의 vanilla PiD 행**보다 좋아야 합니다. 예를 들어 OminiControl canny, K=24에서는 tolerant canny F1이 512 view에서
0.719, native 2048에서 0.669를 넘어야 하고, MUSIQ는 낮지 않고 LPIPS는 나쁘지 않아야 합니다. VAE decode(0.771)를 넘는 것은 필요조건일 뿐이고,
기준은 vanilla PiD입니다. 그래야 "condition을 디코더에 다시 넣은 것"의 효과가 분리됩니다.

## 2. 평가 규칙 (한 번만 읽으면 됩니다)

- **해상도.** 벤치마크의 기준 해상도는 2048이고, 512 view는 거기서 INTER_AREA로 내린 것입니다. 점수는 512 view와 native 2048에서 따로 내고
  같은 해상도 안에서만 비교합니다. 업샘플해서 점수를 내는 일은 없습니다.
- **edge F1.** Canny(100, 200)을 출력에서 다시 뽑고, condition은 nearest로 맞춘 뒤 "condition 픽셀 1개" 안에 들어오면 맞은 것으로 봅니다
  (512에서 1px, 2048에서 4px). strict F1은 512에서만 anchor로 같이 찍습니다. Canny threshold는 해상도에 따라 낮추지 않으므로 2048에서는
  실제로 선명한 edge가 있어야 점수가 납니다. 진짜 이미지를 bicubic으로 키운 것도 0.55밖에 못 받습니다.
- **512에서 2048로 갈 때 F1이 떨어지는 이유는 precision입니다.** dev-200에서 분해해 보면 recall은 떨어지지 않고 오히려 오릅니다
  (PiD K=24: 0.76 -> 0.86). 떨어지는 것은 precision입니다 (0.69 -> 0.57). 즉 native 해상도에서 디코더가 만든 edge의 25~43%가 condition에
  없는 자리에 있습니다. 텍스처, 지어낸 구조입니다. 512 view에서는 area downsample이 그 미세 텍스처를 평균으로 지워 버려서 보이지 않습니다.
  그래서 2048 열은 "condition이 요구하지 않은 디테일을 얼마나 만들어 냈나"를 재는 열이고, CGD가 해야 할 일은 recall을 잃지 않으면서
  2048에서 precision을 올리는 것입니다. 스코어러가 precision / recall / edge density를 이미지별로 기록하니 어느 쪽이 움직였는지 확인하세요.
- **2048 열의 상한은 0.80입니다.** PiD round trip이 그 값이고, 얇은 2048 edge는 512 condition의 4px 블록과 완벽히 겹치지 않기 때문입니다.
  1.0 기준으로 읽지 마세요.
- **‡ (double dagger).** 512만 내는 행(VAE decode 등)의 2048 값은 bicubic x4로 키워서 잰 참고값입니다. 보간으로 2048을 만들면 얼마나
  나오는지 보여주는 것이고, native 출력이 아닙니다.
- **depth, seg, subject**는 스코어러가 내부에서 크기를 맞추므로 해상도에 무관합니다. 512 view 값만 보면 됩니다.
- **c2048 열 (표 A의 세 번째 edge 열).** 2048 출력을 2048 edge map(c2048)에 대해 1px tolerance로 잰 점수이고, 모든 행에 있습니다. 디코더에 2048 condition을 주는 설정은 여기서 강점이 드러납니다. 2048 condition은 진짜 이미지의 PiD round trip에서
  뽑은 Canny이고(`multigen5k/conditions/canny2048`), 그래서 round trip이 기준(1.0)입니다. tolerance는 1px. 여기서 vanilla PiD K=24는
  0.51, 보간 경로는 0.11에서 0.31이므로 condition-aware 디코더의 여지가 가장 큽니다. 생성기 입력은 그대로 512 condition입니다.

## 3. 어떻게 점수를 내나

```
cd cgd-dev200/bench
CUDA_VISIBLE_DEVICES=0 python eval_variant.py --name <내이름_v1> --gen-dir <2048 PNG 폴더> --controller omini --condition canny --k 24 [--subset500]
```

이 한 줄이 512 vs c512, 2048 vs c512, 2048 vs c2048 세 열을 모두 채점하고, `BASELINES.md`의 VAE decode / vanilla PiD 행과의 차이, paired
bootstrap 95% 신뢰구간, 그리고 게이트 PASS / FAIL을 출력합니다 (`results/variants/<이름>/REPORT.md`). 빠르게 보려면 `--subset500`
(500장)으로 먼저 돌리세요. 5000장에서는 F1 차이 0.005면 유의합니다. dev-200 루프(`scripts/03_score.py`)는 그대로 쓰되, 그 숫자는 논문에
쓰지 않습니다.

## 4. CGD 학습 시 condition 규칙

- 학습 타깃은 2048 PiD 디코드이고, condition은 두 형태를 저장합니다: 타깃을 512로 내려서 뽑은 것과 타깃 자체(2048)에서 뽑은 것. 둘을
  반반 섞어 학습하면 한 디코더가 둘 다 받습니다. 디코더 내부에서 512 map을 늘려 써야 하면 edge는 nearest, depth는 bicubic입니다.
- c2048 형태를 위해 `make_targets.py`는 타깃의 2048 Canny(`conditions/canny2048`)도 저장합니다. 학습 때 512 형태와 2048 형태를
  예시마다 반반 섞으면 하나의 디코더가 둘 다 받습니다.
- **CGD는 두 가지 condition으로 모두 평가합니다.** 같은 latent를 512 condition으로 디코딩한 결과(`<이름>_c512`, vanilla PiD와의 동일 조건
  비교)와 2048 condition으로 디코딩한 결과(`<이름>_c2048`, pixel-space 디코더만 쓸 수 있는 입력) 둘 다 내고 둘 다 채점하세요. condition은
  스케일 간에 리사이즈하지 않고 각 스케일에서 다시 뽑은 것입니다 (edge map은 리사이즈하면 망가짐). 2048 condition으로 디코딩한 결과를
  `_c512` 행으로 표기하지만 마세요.

## 5. PiD는 두 가지입니다

논문 본문의 PiD 행은 공개된 4-step distilled student입니다. CGD는 undistilled teacher (`PiD_v1pt5_res2kto4k_sr4x_official_flux_undistilled`,
25 steps, CFG 5) 위에 학습하므로, 정확한 ablation은 teacher 행(`pidt_k<K>`)과의 비교입니다. teacher 행은 subset500에서 student와 짝지어
계산 중이며 (`BASELINES.md` 표 F, 2026-09-13 새벽 완료 예정), `decode_pid.py --pid-ckpt-type teacher`로 재현할 수 있습니다.
`eval_variant.py --teacher`로 teacher 행과도 비교됩니다.

## 6. 환경 주의

컨테이너 `wookiekim_tfso`에서 `pip install`은 반드시 `--no-deps`로 하세요. 2026-09-12에 일반 설치가 torch를 2.14로 올려서 새 프로세스가 전부
깨졌고, 2.5.1+cu121로 복구했습니다.
