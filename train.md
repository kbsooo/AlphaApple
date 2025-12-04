# FruitBox Maskable PPO 학습 가이드

`train/train_maskable_ppo.py` 스크립트가 어떻게 FruitBox 환경에서 Maskable PPO 에이전트를 학습시키는지, 그리고 실험을 커스터마이즈하는 방법을 정리했습니다.

## 파이프라인 개요

1. **환경 설정** — `FruitBoxConfig`로 보드 크기, 에피소드 최대 스텝, 한 칸 제거당 보상을 제어합니다.
2. **래퍼 구성**
   - `FloatChannelObs`: `(rows, cols)` 정수 보드를 정규화된 `(1, rows, cols)` float32 텐서로 변환하여 CNN 정책 입력과 맞춥니다.
   - `ActionMasker`: 환경이 반환하는 합법 행동 마스크(`info['action_mask']`)를 사용해 잘못된 직사각형이 샘플되지 않도록 합니다.
3. **특징 추출기** — 스크립트 내부의 `SmallGridCNN`은 3개의 얕은 컨볼루션과 선형 계층으로 128차원 잠재벡터를 만들어 정책/가치 헤드가 공유하도록 합니다.
4. **벡터화된 수집** — `make_vec_env`가 기본적으로 `SubprocVecEnv`를 구성하고, `--no-parallel` 옵션을 주면 `DummyVecEnv`로 전환합니다. 각 인스턴스는 `Monitor`로 감싸 에피소드 통계를 기록합니다.
5. **Maskable PPO 에이전트** — 커스텀 CNN과 SB3 기본 정책 헤드를 결합해 격자 기반 과제에 적합한 기본 하이퍼파라미터로 초기화합니다.
6. **콜백** — 체크포인트 저장과 주기적 평가를 CLI 플래그로 제어할 수 있으며, 평가는 `MaskableEvalCallback`을 사용해 행동 마스크를 존중합니다.

## 학습 실행 방법

```bash
uv run python train/train_maskable_ppo.py \
    --rows 10 --cols 17 \
    --total-timesteps 1000000 \
    --n-envs 8 \
    --log-dir logs \
    --ckpt-dir ckpts
```

생성물:

- `--ckpt-dir` 아래에 주기적인 체크포인트가 저장되며, 최종 모델은 `fruitbox_ppo_final.zip`, 최고 성능 모델은 `best/`에 저장됩니다.
- 각 환경 모니터 CSV가 `--log-dir`에 기록됩니다.
- 평가를 활성화하면 `--log-dir/eval_metrics`에 평가 결과가 쌓입니다.
- TensorBoard 로그는 `--tensorboard` 경로(기본 `tb/`)에 위치합니다.

## 주요 CLI 인자

| 플래그 | 설명 | 기본값 |
| --- | --- | --- |
| `--rows`, `--cols` | 보드 크기 | `10`, `17` |
| `--max-steps` | 에피소드 최대 스텝 수 | `500` |
| `--reward-per-cell` | 한 칸 제거당 보상 계수 | `1.0` |
| `--total-timesteps` | 전체 학습 스텝 예산 | `1_000_000` |
| `--n-envs` | 병렬 환경 수 | `8` |
| `--learning-rate`, `--gamma`, `--n-steps`, `--batch-size`, `--ent-coef`, `--clip-range` | PPO 하이퍼파라미터 | 스크립트 기본값 |
| `--checkpoint-freq` | 체크포인트 저장 주기(타임스텝) | `100_000` |
| `--eval-freq`, `--eval-episodes` | 평가 주기와 에피소드 수 | `100_000`, `10` |
| `--no-parallel` | 단일 프로세스 환경 사용(디버깅용) | 비활성 |

모든 옵션은 `uv run python train/train_maskable_ppo.py --help`로 확인할 수 있습니다.

## 커스터마이즈 팁

- **특징 차원 확대** — `--policy-features-dim` 값을 키우거나 `SmallGridCNN` 구조를 확장해 모델 용량을 늘려보세요. TensorBoard로 과적합 여부를 모니터링하는 것을 권장합니다.
- **보상 조정** — 스텝 패널티나 종료 보너스를 추가하려면 `FruitBoxEnv`를 수정한 뒤 `reward_per_cell`과 하이퍼파라미터를 함께 재조정합니다.
- **커리큘럼 학습** — 여러 보드 크기를 순차적으로 학습시키는 스크립트를 작성하거나, 환경에서 난이도 변화를 지원하도록 확장해볼 수 있습니다.
- **평가 정책 제어** — 기본값은 결정론적 정책(`deterministic=True`)입니다. 확률적 평가가 필요하면 콜백 생성 시 `deterministic=False`로 변경하세요.
- **디바이스 선택** — `--device cpu`, `--device cuda`, `--device cuda:1`과 같이 PyTorch 디바이스를 명시적으로 지정할 수 있습니다.

## 다음 단계 제안

- 체크포인트를 불러와 평균 리턴을 계산하는 간단한 `evaluate.py` 스크립트를 추가합니다.
- 학습된 정책을 ONNX로 내보내어 웹 추론 파이프라인과 연동합니다.
- 실험 재현성을 위해 하이퍼파라미터, Git 커밋 해시 등을 별도 로그 파일에 저장하는 방식을 도입합니다.

