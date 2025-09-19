# 🍎 AlphaApple - FruitBox RL (WIP)

사과게임(FruitBox) 환경을 Gymnasium 스타일로 구현하고, 대규모 이산 행동공간(모든 직사각형 선택)에 대해 Maskable PPO로 학습하는 계획을 담은 리포지토리입니다.

현재 코드 베이스는 환경 구현이 중심이며, 학습/평가 스크립트는 아래 가이드를 따라 쉽게 추가할 수 있습니다.

## 📦 구성 요소

- `envs/fruitbox_env.py`: FruitBox 게임 환경 구현
- `pyproject.toml`: 의존성과 패키징 메타데이터

## 🧩 환경 개요

- 관찰: `rows×cols` 정수 격자(기본 10×17), 값 0~9. 0은 빈칸입니다.
- 행동: 모든 축정렬 직사각형 `(r1,c1,r2,c2)`을 미리 열거한 `Discrete(N)`.
- 마스킹: 합이 정확히 10이고, 0을 포함하지 않는 직사각형만 합법(True)으로 표시됩니다. 마스크는 `reset/step`의 `info['action_mask']`로 제공됩니다.
- 종료: 합법 행동이 더 이상 없거나, 안전 상한 `max_steps` 도달 시 종료.
- 보상: 선택한 직사각형 넓이(셀 수) × `reward_per_cell`.

주의: 기본 보드 크기는 `rows=10, cols=17`입니다. 실제 게임 보드가 17×10이면 설정을 바꿔 사용하세요.

## 🧠 권장 학습 접근

- 알고리즘: `sb3-contrib`의 MaskablePPO (불법 행동을 확률분포에서 제거)
- 전처리: 관찰을 0~1로 정규화하고 채널 차원(1×R×C)을 추가하는 래퍼
- 특징추출: 작은 CNN(SmallGridCNN)으로 128차원 특징 벡터 추출
- 병렬 수집: 8개 내외 병렬 환경(`SubprocVecEnv`) 권장
- 시작 하이퍼파라미터: `lr=3e-4, γ=0.995, clip=0.2, n_steps≈8k, batch_size=64~256, ent_coef≈0.01`

## 🚀 빠른 시작

### 의존성 설치

```bash
uv install
# 또는
pip install -e .
```

### 필수 래퍼들

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces, ObservationWrapper

class FloatChannelObs(ObservationWrapper):
    """(R,C) int8 → (1,R,C) float32 in [0,1]"""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        R, C = env.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1, R, C), dtype=np.float32)

    def observation(self, obs):
        return (obs.astype(np.float32) / 9.0)[None, ...]

# sb3-contrib의 ActionMasker 사용 시, 합법 행동 마스크를 제공
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[env.legal_actions()] = True
    return mask
```

### SmallGridCNN (SB3 특징추출기)

```python
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SmallGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_ch, H, W = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(obs))
```

### 최소 학습 예제 (스크립트/노트북용)

```python
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.fruitbox_env import FruitBoxEnv, FruitBoxConfig

def make_env(seed):
    def _thunk():
        env = FruitBoxEnv(FruitBoxConfig(rows=10, cols=17))
        env = FloatChannelObs(env)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed)
        return env
    return _thunk

n_envs = 8
vec_env = SubprocVecEnv([make_env(42 + i) for i in range(n_envs)])

policy_kwargs = dict(
    features_extractor_class=SmallGridCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

model = MaskablePPO(
    "CnnPolicy",
    vec_env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    gamma=0.995,
    n_steps=2048,   # 2048 × 8env ≈ 16k/rollout
    batch_size=256,
    ent_coef=0.01,
    verbose=1,
)

model.learn(total_timesteps=1_000_000)
model.save("ckpts/fruitbox_ppo.zip")
```

### 평가 예제

```python
env = ActionMasker(FloatChannelObs(FruitBoxEnv()), mask_fn)
obs, info = env.reset(seed=0)
total = 0.0
while True:
    mask = info.get("action_mask")
    action, _ = model.predict(obs, deterministic=True, action_masks=mask)
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    if terminated or truncated:
        break
print("Return:", total)
```

## 📁 현재 구조

```
alphaapple/
├── envs/fruitbox_env.py
├── pyproject.toml
└── README.md
```

## 🧭 로드맵 / TODO

- [ ] 학습 스크립트 `train/train_maskable_ppo.py` 추가
- [ ] 평가/시각화 도구 추가 (`evaluate.py`, TensorBoard 설정)
- [ ] ONNX 내보내기/테스트 스크립트 추가
- [ ] 커리큘럼 학습(작은 보드 → 큰 보드) 실험
- [ ] 모델 경량화(양자화/프루닝)와 웹 배포 예제

## 참고

- 본 환경의 보상은 스텝 패널티/완료 보너스를 포함하지 않습니다. 필요 시 환경을 확장하거나 하이퍼파라미터로 보완하세요.
- 행동 마스크는 환경이 계산하여 `info['action_mask']`로 제공하며, 학습 시 `ActionMasker`로 정책에 반영합니다.

1. Fork 후 feature branch 생성
2. 변경사항 커밋
3. Pull Request 생성

## 📄 라이선스

MIT License

## 📞 문의

- **HuggingFace:** https://huggingface.co/kbsooo/AlphaApple
- **Issues:** GitHub Issues 활용
- **모델 사용 문의:** HuggingFace 모델 페이지 코멘트

---

**Made with 🍎 by kbsooo | Powered by PPO & ONNX**
