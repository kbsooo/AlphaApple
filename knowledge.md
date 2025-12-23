# AlphaApple - FruitBox RL 프로젝트

## 프로젝트 개요

FruitBox 게임을 강화학습으로 해결하는 프로젝트. 10x17 보드에서 합이 10인 사각형 영역을 찾아 제거하는 게임.

- **보드 크기**: 10 x 17 = 170 셀
- **Action Space**: 8,415개 (가능한 모든 사각형)
- **목표**: 최대한 많은 사과(셀) 제거

---

## 현재 상태 진단 (2024-12)

### 성능 요약

| 정책 | Reward | 제거율 |
|------|--------|--------|
| DQN (BackwardGenerator 학습) | 96.0 | 56.5% |
| DQN (Random 보드 학습) | 93.0 | 54.7% |
| max_reward 휴리스틱 | **97.0** | 57.1% |
| random | 91.0 | 53.5% |

**결론**: 학습된 DQN이 단순 휴리스틱보다 못함.

---

## 발견된 문제점

### 1. 모델 아키텍처 문제: Activation Explosion

```
Layer별 Activation 크기:
- conv1 출력: mean = 2.9      (정상)
- conv2 출력: mean = 218      (10배 증가)
- conv3 출력: mean = 32,126   (100배 폭발)
- fc1 입력:   sum = 3.5억
- Q-value:    range = [-10억, +1억]  (의미 없는 값)
```

**원인**: Conv layer 사이에 **BatchNormalization이 없음**

**결과**:
- Q-value가 비현실적인 값 (이론상 최대 ~170인데 -10억)
- 상대적 순서만으로 간신히 작동 → random보다 약간 나은 수준

**현재 모델 구조** (`src/models.py`):
```python
class FruitBoxDQN(nn.Module):
    def __init__(self, rows, cols, n_actions):
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # BatchNorm 없음!
        self.fc1 = nn.Linear(64 * rows * cols, 512)
        self.fc2 = nn.Linear(512, n_actions)
```

---

### 2. 학습 데이터 문제: Distribution Shift

#### BackwardBoardGenerator vs 실제 보드 비교

| 항목 | BackwardGenerator (학습) | 실제 보드 (테스트) |
|------|--------------------------|-------------------|
| 합=10인 영역 수 | **124개** | **46개** |
| 숫자 분포 | 1, 2 위주 (작은 수) | 1-9 균등 분포 |
| 설계 의도 | 해답이 보장되도록 역설계 | 완전 랜덤 |
| 난이도 | 쉬움 | 어려움 |

**BackwardGenerator 동작 방식**:
1. 빈 보드에서 시작
2. 랜덤 사각형 선택 → 합=10이 되도록 숫자 배치
3. 반복 → "해답이 풍부한" 보드 생성
4. 결과: 작은 숫자(1,2,3)가 많고, 합=10 조합이 쉽게 나옴

**결과**: 모델이 "쉬운 보드 패턴"만 학습 → 실제 보드에서 일반화 실패

---

### 3. 알고리즘 한계: 장기 계획 학습의 어려움

#### 게임의 본질

```
현재: [7] [3] [5] [2]    가능한 조합: [3,5,2]=10
      [4] [6] [8] [1]

[3,5,2] 제거 후:
      [7] [0] [0] [0]    새로운 가능성: [7,0,0,0,1,0,0,2]=10
      [4] [6] [8] [1]    → 훨씬 큰 영역 제거 가능!
```

**빈 공간(0)은 "미래의 자산"** - 지금 최선이 아니어도, 나중에 큰 영역을 제거할 수 있게 함.

#### DQN이 이걸 학습하기 어려운 이유

1. **Credit Assignment 문제**
   - Step 1의 작은 제거가 Step 10의 큰 보상을 가능하게 함
   - Q-learning은 이 연결을 학습하기 어려움

2. **Greedy Reward 함정**
   - 현재 보상: `제거한 사과 수`
   - 모델이 학습하는 것: "지금 당장 많이 제거"
   - 좋은 전략: "지금 조금, 나중에 크게"

3. **탐색 공간 폭발**
   - 한 스텝에 ~50개 valid action
   - 10스텝 계획: 50^10 = 천문학적 경우의 수
   - DQN은 이걸 암기하려 함 → 불가능

---

## 개선 방안

### Phase 1: 모델 안정화 (필수)

**목표**: Activation 폭발 해결, 의미 있는 Q-value 출력

```python
class FruitBoxDQN_v2(nn.Module):
    def __init__(self, rows, cols, n_actions):
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 추가
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 추가
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # 추가
        self.fc1 = nn.Linear(64 * rows * cols, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

**예상 효과**: Q-value가 현실적인 범위 (0~200)로 수렴

---

### Phase 2: 학습 데이터 다양화

**옵션 A: 혼합 학습**
```python
# 50% BackwardGenerator + 50% Random
if random.random() < 0.5:
    board = backward_generator.generate()
else:
    board = random_board()
```

**옵션 B: 점진적 전환**
```python
# Episode 진행에 따라 BackwardGenerator 비율 감소
backward_ratio = max(0.2, 1.0 - episode / 10000)
```

**옵션 C: 도메인 랜덤화**
- 숫자 분포 변경 (균등, 치우침, 희소 등)
- 보드 크기 변경 (작은 보드 → 큰 보드)

---

### Phase 3: 보상 설계 개선

**현재**: `reward = 제거한_사과_수`

**개선안 1: 미래 가능성 보너스**
```python
new_valid_actions = count_valid_actions(new_board)
old_valid_actions = count_valid_actions(old_board)
future_bonus = 0.1 * (new_valid_actions - old_valid_actions)
reward = cleared_cells + future_bonus
```

**개선안 2: 영역 크기 보너스**
```python
area_bonus = 0.5 * (area - 2)  # 큰 영역 선호
reward = cleared_cells + area_bonus
```

**개선안 3: 게임 종료 보너스**
```python
if game_cleared:
    reward += 50  # 완전 클리어 보너스
```

---

### Phase 4: 알고리즘 변경 (장기)

#### 옵션 A: MCTS + Neural Network (AlphaZero 스타일)

```
장점:
- 명시적인 탐색으로 장기 계획 가능
- 학습 없이도 어느 정도 작동
- 학습과 탐색의 시너지

단점:
- 구현 복잡
- 추론 시 계산량 많음
```

#### 옵션 B: PPO + Intrinsic Motivation

```
장점:
- 탐색 보상으로 다양한 전략 발견
- DQN보다 안정적인 학습

단점:
- 하이퍼파라미터 튜닝 필요
```

#### 옵션 C: Model-based RL (Dreamer 등)

```
장점:
- 환경 모델을 학습 → 시뮬레이션으로 계획
- 샘플 효율성 높음

단점:
- 환경 모델 정확도가 핵심
```

---

## 우선순위 권장

```
1. [긴급] BatchNorm 추가하여 모델 안정화
2. [중요] 혼합 학습으로 Distribution Shift 해결
3. [선택] 보상 설계 실험
4. [장기] MCTS 또는 PPO로 알고리즘 변경
```

---

## 파일 구조

```
alphaapple/
├── envs/
│   ├── fruitbox_env.py      # 게임 환경
│   └── backward_generator.py # 보드 생성기
├── src/
│   ├── models.py            # DQN 모델 (BatchNorm 필요)
│   └── agent.py             # DQN 에이전트
├── experiments/
│   ├── train_colab_integrated.py  # 통합 학습 스크립트
│   ├── train_baseline_mps.py      # MPS 학습 스크립트
│   └── play_board.py              # 테스트 스크립트
├── checkpoints/
│   ├── dqn_colab_final.pt         # BackwardGenerator 학습 모델
│   └── dqn_no_backward_final.pt   # Random 보드 학습 모델
├── extension/                     # Chrome 확장 프로그램
└── board.py                       # 테스트용 보드
```

---

## 참고: 테스트 명령어

```bash
# 모델 테스트 (여러 정책 비교)
uv run python experiments/play_board.py \
    --model_path checkpoints/dqn_colab_final.pt \
    --board_path board.py \
    --compare \
    --render_every 0

# 특정 정책만 테스트
uv run python experiments/play_board.py \
    --model_path checkpoints/dqn_colab_final.pt \
    --board_path board.py \
    --policy q \
    --render_every 1
```
