# 🍎 AlphaApple - RL for Perfect FruitBox Play

**목표**: 사과게임(FruitBox) 170개 셀 **전부 제거** (100% 클리어)

강화학습으로 인간을 넘어서는 성능 달성을 목표로 하는 프로젝트입니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kbsooo/AlphaApple/blob/main/train_colab.ipynb)

## 🎯 성능 현황

```
┌────────────────────────────────────────────┐
│ 벤치마크                                    │
├────────────────────────────────────────────┤
│ 사람 최고:      130개 (76.5%)             │
│ Greedy 전략:    105개 (62.0%)             │
│ V2 모델:        104개 (61.3%) [95% 보드]  │
│ V3 모델:        115-125개 예상 (67-74%)   │
│                                            │
│ 최종 목표:      170개 (100%) ⭐           │
└────────────────────────────────────────────┘
```

**최신 업데이트 (V3)**:
- ✅ Beam Search로 고품질 데이터 수집
- ✅ Data Augmentation (회전/대칭)
- ✅ Early Stopping으로 overfitting 방지

## 💡 핵심 아이디어

### 1️⃣ Autoregressive Policy
8,415개의 거대한 행동 공간을 **4단계 순차 선택**으로 분해:
```
r1 선택 (10개) → c1 선택 (17개) → r2 선택 (10개) → c2 선택 (17개)
```
- 행동 공간: 8,415 → 54로 효율화
- 자연스러운 마스킹 (r2≥r1, c2≥c1)
- 좌표 간 의존성 명시적 모델링

### 2️⃣ 경량 모델 설계
**큰 모델 ≠ 좋은 모델**
- **706K parameters** (기존 2.9M 대비 75.6% 축소)
- 3배 빠른 학습 속도
- Colab 무료 GPU에서 30분 만에 학습 완료
- 과적합 방지, 일반화 성능 향상

### 3️⃣ 역방향 보드 생성
랜덤 보드는 **최대 45%만 제거 가능** → 학습 의미 없음
```python
# 해결: 제거 가능한 보드만 생성
generator = BackwardBoardGenerator(rows=10, cols=17)
board, solution = generator.generate(target_coverage=0.95)
# → 95% 제거 보장!
```

## 🚀 빠른 시작 (Colab 권장)

### Colab에서 학습

**V4 (최신, 권장)**: Expert Iteration (Self-Improvement)
- 학습 시간: ~60-90분 (5 iterations)
- 예상 성능: 110-125개 (64-73%)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kbsooo/AlphaApple/blob/main/train_colab_v4.ipynb)

**V3**: Multiple Rollouts
- 학습 시간: ~30-60분
- 결과: 101.9개 (59.9%) - V2와 유사
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kbsooo/AlphaApple/blob/main/train_colab_v3.ipynb)

**V2**: Action Mask 기반 학습
- 학습 시간: ~10분
- 결과: 101.5개 (59.7%)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kbsooo/AlphaApple/blob/main/train_colab_v2.ipynb)

**단계**:
1. 위 배지 클릭
2. 런타임 → GPU로 변경
3. 전체 셀 실행
4. `bc_policy_best_vX.pt` 다운로드

**완료!** 사람 수준에 근접한 성능의 모델을 얻습니다.

### 로컬 설치

```bash
# 의존성 설치
pip install gymnasium numpy torch tqdm

# 코드 다운로드
git clone https://github.com/kbsooo/AlphaApple.git
cd AlphaApple
```

### 환경 테스트

```python
from envs.fruitbox_env import FruitBoxEnv, FruitBoxConfig

env = FruitBoxEnv(FruitBoxConfig(rows=10, cols=17))
obs, info = env.reset(seed=42)

print(f"보드 크기: {obs.shape}")
print(f"행동 공간: {env.action_space.n}")
print(f"초기 합법 행동: {sum(info['action_mask'])}개")
```

### 학습된 모델 평가

```python
import torch
from models.lightweight_policy import LightweightPolicy
from scripts.evaluate import evaluate_policy

# 모델 로드
policy = LightweightPolicy(rows=10, cols=17)
policy.load_state_dict(torch.load('bc_policy_best.pt'))

# 평가
results = evaluate_policy(policy, n_episodes=100)
print(f"평균 성능: {results['mean_reward']:.1f}개")
```

## 📦 프로젝트 구조

```
AlphaApple/
├── envs/
│   ├── fruitbox_env.py              # 게임 환경 (0 포함 허용)
│   ├── backward_generator.py        # 역방향 보드 생성
│   └── autoregressive_wrapper.py    # 좌표→행동 변환
│
├── models/
│   ├── autoregressive_policy.py     # 기본 모델 (2.9M params)
│   └── lightweight_policy.py        # 경량 모델 (706K params) ⭐
│
├── scripts/
│   ├── collect_expert_data.py       # 전문가 데이터 수집
│   ├── train_behavior_cloning.py    # BC 학습
│   └── evaluate.py                  # 성능 평가
│
├── train_colab.ipynb                # Colab 학습 (V1)
├── train_colab_v2.ipynb             # Colab 학습 (V2 - Action Mask)
├── train_colab_v3.ipynb             # Colab 학습 (V3 - Multiple Rollouts)
├── train_colab_v4.ipynb             # Colab 학습 (V4 - Expert Iteration) ⭐
└── README.md
```

## 🎮 게임 규칙

**FruitBox (사과게임)**:
- 10×17 격자에 1-9 숫자 배치
- 직사각형 선택 → 합이 정확히 10이면 제거
- **0(빈 칸)을 포함해도 합=10이면 OK**
- 더 이상 제거 불가능하면 게임 종료

**목표**: 170개 셀 전부 제거

## 🧠 학습 전략

### Phase 1: Behavior Cloning (구현 완료)
```python
# 1. 역방향 생성으로 95% 제거 가능한 보드 생성
# 2. "작은 것 우선" 전략으로 전문가 데이터 수집
# 3. 경량 모델을 지도학습으로 사전학습
```
**예상 성능**: 115-120개 (67-70%)

### Phase 2: Expert Iteration (V4 - 현재 진행 중)
```python
# Self-Improvement Loop
# 1. 현재 정책으로 데이터 수집
# 2. 고품질 에피소드만 선택 (top 50%)
# 3. 정책 업데이트
# 4. 반복 (3-5 iterations)
```
**핵심 아이디어**:
- V3 문제: Expert가 약함 (103.7개) → 학습 ceiling
- V4 해결: 정책이 스스로를 가르침 → 점진적 개선
- AlphaGo Zero의 핵심 메커니즘

**예상 성능**: 110-125개 (64-73%)

### Phase 3: PPO Fine-tuning (계획)
```python
# BC 모델을 warm-start로 사용
# PPO로 추가 학습
# Reward shaping: 미래 가능성 고려
```
**예상 성능**: 130-145개 (76-85%)

### Phase 4: MCTS + RL (선택사항)
```python
# 완벽한 플레이를 위한 고급 기법
# AlphaZero 스타일 MCTS
```
**예상 성능**: 160-170개 (94-100%) 🎯

## 📊 실험 결과

### 버전별 성능 비교

| 버전 | 방법 | 평균 (95% 보드) | 평균 (일반) | 최대 | 비고 |
|------|------|----------------|------------|------|------|
| V1 | BC (no mask) | -500 (0%) | -500 (0%) | 0 | 불법 행동 문제 |
| V2 | BC + Action Mask | 101.5 (59.7%) | 104.2 (61.3%) | 129 | Greedy 수준 |
| **V3** | **Multiple Rollouts** | **101.9 (59.9%)** | **103.7 (61.0%)** | **120** | **V2와 유사, 큰 개선 없음** |

### 베이스라인 비교

| 전략 | 평균 제거 | 범위 | 비고 |
|------|-----------|------|------|
| Random | 99.7개 (58.6%) | [86-119] | 순전히 랜덤 |
| Greedy (큰 것) | 96.4개 (56.7%) | [78-114] | 직관적이지만 나쁨 |
| **Greedy (작은 것)** | **105.4개 (62.0%)** | [91-130] | 최고 휴리스틱 |
| V2 모델 (BC) | 104.2개 (61.3%) | [84-129] | Greedy와 비슷 |
| Beam Search (width=20) | 110-120개 (예상) | - | V3 데이터 수집용 ⭐ |

**핵심 발견**:
1. "작은 것 우선"이 "큰 것 우선"보다 9개 더 좋음
   - 이유: 작은 조합 제거 → 0 생성 → 미래 자원
2. V2는 Greedy 수준에서 정체 → 더 나은 데이터 필요
3. **V3 분석 (2025-11-10)**:
   - Multiple Rollouts로 데이터 다양성 증가 시도
   - 결과: V2와 거의 동일 (101.9 vs 101.5)
   - 평균 Expert 보상: 103.7개 (학습 ceiling)
   - **문제**: Expert 자체가 약함 (103.7개) → 학습 한계
   - **해결**: Expert Iteration으로 Self-Improvement 필요

### 환경 수정의 영향

```
┌────────────────────────────────────────────┐
│ 규칙 수정 전후                              │
├────────────────────────────────────────────┤
│ 이전 (0 불가):     71개 (41.9%)           │
│ 수정 (0 가능):    105개 (62.0%)           │
│                                            │
│ 개선:            +34개 (+48%)              │
└────────────────────────────────────────────┘
```

**Critical Bug Fix**: 실제 게임은 `[3, 0, 7]` 같은 0 포함 조합도 합=10이면 가능!

## 🔬 기술적 세부사항

### 모델 아키텍처

```python
class LightweightPolicy(nn.Module):
    def __init__(self):
        # Board Encoder: CNN (16→32 channels)
        self.board_encoder = LightweightBoardEncoder(latent_dim=128)

        # Coordinate Embeddings (16-dim)
        self.r_embed = nn.Embedding(10, 16)
        self.c_embed = nn.Embedding(17, 16)

        # Shared Decoders (모든 좌표에 재사용)
        self.row_decoder = nn.Linear(128 + 32, 10)
        self.col_decoder = nn.Linear(128 + 32, 17)

        # Value Head (PPO용)
        self.value_head = nn.Linear(128, 1)
```

**총 파라미터**: 706,156개

### 학습 하이퍼파라미터

```python
# Behavior Cloning
optimizer = Adam(lr=3e-4)
batch_size = 128
epochs = 30
train/val split = 90/10
```

### 데이터 수집

```python
# 역방향 생성 보드
n_episodes = 500
target_coverage = 0.95  # 95% 제거 가능
strategy = "small_first"  # 작은 것 우선
```

## 🎓 학습한 교훈

### 1. 환경 구현이 가장 중요
잘못된 규칙 (0 불가) → 41.9% 성능
올바른 규칙 (0 가능) → 62.0% 성능
**+48% 향상!**

### 2. 큰 모델이 답이 아니다
2.9M params → 과적합, 느린 학습
706K params → 빠른 수렴, 좋은 일반화

### 3. 보드 생성이 성능 천장 결정
랜덤 보드: 최대 45% (학습 무의미)
역방향 보드: 최대 95% (학습 의미 있음)

### 4. 직관과 반대되는 전략이 더 좋을 수 있다
"큰 것 우선" (직관적) → 96.4개
"작은 것 우선" (반직관) → 105.4개

## 📈 로드맵

- [x] 환경 구현 (0 포함 허용)
- [x] 역방향 보드 생성기
- [x] Autoregressive Policy
- [x] 경량 모델 (706K params)
- [x] Behavior Cloning 파이프라인 (V1)
- [x] Action Mask 기반 학습 (V2)
- [x] Multiple Rollouts (V3) - 결과: V2와 유사
- [x] Colab 학습 노트북 (V1/V2/V3)
- [x] V3 실험 결과 확인 (2025-11-10)
- [ ] **Expert Iteration (V4)** ← 현재 진행 중
- [ ] PPO Fine-tuning
- [ ] MCTS 통합
- [ ] ONNX 변환 & 웹 배포
- [ ] **170점 만점 달성** 🎯

## 🤝 기여

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

MIT License

## 📞 문의

- **HuggingFace**: https://huggingface.co/kbsooo/AlphaApple
- **GitHub Issues**: [Report bugs or request features](https://github.com/kbsooo/AlphaApple/issues)
- **Author**: kbsooo

## 📝 V4 Expert Iteration 상세 설명

### 왜 Expert Iteration인가?

**V3의 문제점**:
- Multiple Rollouts로 데이터 다양성 증가 시도
- 결과: V2와 거의 동일 (101.9개 vs 101.5개)
- **근본 원인**: Expert가 약함 (평균 103.7개) → 학습 ceiling

**V4의 해결책**: Self-Improvement Loop
```
for iteration in range(5):
    1. 현재 정책으로 500개 보드 플레이 (각 5 rollouts)
    2. 고품질 에피소드만 선택 (top 50%)
    3. 선택된 데이터로 정책 학습
    4. 평가 → 개선 확인
```

### 왜 작동하는가?

**점진적 개선 메커니즘**:
```
Iteration 0: 초기 정책 (100개)
  → 500개 보드 플레이
  → 운 좋게 일부 보드에서 105개 달성
  → Top 50% 선택 (평균 105개)
  → 학습

Iteration 1: 개선된 정책 (105개)
  → 이제 안정적으로 105개 달성
  → 일부 보드에서 110개 달성
  → Top 50% 선택 (평균 110개)
  → 학습

Iteration N: 계속 개선...
```

**AlphaGo Zero의 핵심 메커니즘**:
- 자기 자신과 플레이 (self-play)
- 좋은 플레이만 학습 (selective learning)
- 점진적 성능 향상 (iterative improvement)

### 구현 세부사항

**데이터 수집 (collect_self_play_data)**:
- 현재 정책으로 각 보드 5번 플레이
- 다양성 확보: deterministic=False로 stochastic sampling
- Best rollout만 저장
- Top 50% 선택 (threshold 기반도 가능)

**학습 (train_on_data)**:
- 선택된 데이터로 Behavior Cloning
- 15 epochs (빠른 학습)
- Train/Val split: 90/10
- Early stopping 없음 (데이터가 계속 바뀌므로)

**평가 (evaluate_policy_quick)**:
- 빠른 평가: 20-30 episodes
- 95% 제거 가능 보드 사용
- Deterministic 정책 평가

### 하이퍼파라미터

```python
N_ITERATIONS = 5              # 반복 횟수
N_EPISODES_PER_ITER = 500     # iteration당 보드 수
N_ROLLOUTS = 5                # 보드당 플레이 횟수
TOP_K_PCT = 0.5               # 상위 50% 선택
N_EPOCHS_PER_ITER = 15        # iteration당 학습 epochs
```

### 다음 세션 참고사항

1. **V4 실행 시간**: ~60-90분 (Colab T4 GPU 기준)
2. **V3 모델 warm-start**: 있으면 로드, 없으면 처음부터 학습
3. **모델 저장**: 각 iteration마다 `bc_policy_v4_iter{N}.pt` 저장
4. **평가**: 각 iteration 후 30 episodes로 빠른 평가
5. **최종 평가**: 최고 성능 iteration 선택 후 50 episodes로 평가

### 예상 결과

- **Best Case**: 110-125개 (64-73%)
- **Typical Case**: 105-115개 (61-67%)
- **Worst Case**: V3와 유사 (101-105개)

**성공 지표**:
- Iteration별로 평균/최대 보상 증가 추세
- 마지막 iteration이 처음보다 5개 이상 개선

### 다음 단계 (V4 이후)

**만약 V4가 115개 이상 달성**:
→ PPO Fine-tuning 시도 (Phase 3)

**만약 V4가 110개 미만**:
→ 더 많은 iterations (10회) 또는 더 큰 데이터셋 (1000 episodes) 시도

**최종 목표**:
→ **170점 만점 달성** 🎯

## 🙏 감사의 말

- OpenAI Gymnasium for the RL framework
- PyTorch for deep learning
- Colab for free GPU resources

---

**Made with 🍎 by kbsooo | Powered by Autoregressive RL**
