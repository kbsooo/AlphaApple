# train_maskable_ppo.py
"""
Fruit Box(사과게임) 환경을 PPO로 학습시키는 스크립트.
- 관측(보드)을 (H, W, 1) float32 [0,1] 범위로 변환
- 작은 CNN(SmallGridCNN)으로 특징 추출
- PPO로 학습 및 주기적 평가·체크포인트 저장
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO

from envs.fruitbox_env import FruitBoxEnv, FruitBoxConfig

# -----------------------------------------------------------------------------
# Custom CNN feature extractor for small grids
# -----------------------------------------------------------------------------
class SmallGridCNN(BaseFeaturesExtractor):
    """(H, W, C) 형태의 작은 격자 보드를 위한 간단한 CNN 특징 추출기.

    Stable-Baselines3의 BaseFeaturesExtractor를 상속하여 policy의 백본으로 사용됨.
    출력 벡터 크기는 features_dim으로 지정(이 스크립트에서는 128).
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # BaseFeaturesExtractor 초기화 (features_dim은 최종 선형층 출력 크기)
        super().__init__(observation_space, features_dim)
        
        # 관측은 (height, width, channels) = (17, 10, 1)로 가정
        n_input_channels = observation_space.shape[2]
        
        # 간단한 CNN 스택: Conv → ReLU × 2, 다운샘플 Conv(stride=2) → ReLU → Flatten
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # downsample
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Flatten 이후의 차원을 한 번의 더미 forward로 계산하여 선형층 입력 크기 산출
        with torch.no_grad():
            # observation_space.sample() → (H, W, C)
            # SB3는 (B, H, W, C) 텐서를 주므로 NCHW로 permute 필요
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float().permute(0, 3, 1, 2)
            ).shape[1]
        
        # 최종 특징 차원을 features_dim으로 맞추는 선형층
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) 입력을 (B, C, H, W)로 변환 후 CNN 통과 → 선형층.
        정책/가치망이 사용할 고정 길이 특징 벡터를 반환.
        """
        # Input shape: (batch, height, width, channel) → (batch, channel, height, width)
        observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))

# -----------------------------------------------------------------------------
# Observation wrappers: 관측을 PPO가 쓰기 좋게 전처리
# -----------------------------------------------------------------------------
class ToFloat01(gym.ObservationWrapper):
    """보드의 정수 값(0~9)을 float32(0.0~1.0)로 정규화."""
    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape
        # 정규화 후 관측 공간 정의 (dtype=float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(h, w), dtype=np.float32)
    def observation(self, obs):
        # 0~9 → 0.0~1.0로 스케일링
        return (obs.astype(np.float32) / 9.0)

class AddChannel(gym.ObservationWrapper):
    """(H, W)을 (H, W, 1)로 채널 차원 추가."""
    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape
        # 채널 1개를 가진 3D 텐서 관측 공간
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(h, w, 1), dtype=np.float32)
    def observation(self, obs):
        # (H, W) → (H, W, 1)
        return obs[..., None]  # (H, W, 1)

# -----------------------------------------------------------------------------
# Action filtering wrapper: 무효 행동이 들어오면 자동으로 유효 행동으로 치환
# -----------------------------------------------------------------------------
class ActionFilter(gym.ActionWrapper):
    """Convert invalid actions to valid ones automatically.

    MaskablePPO를 쓰지 않고 일반 PPO를 쓸 때, 무효 행동을 피하기 위한 안전장치.
    주어진 action이 불법이면, 현재 상태에서 계산한 action mask로 첫 유효 action을 선택.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def action(self, action):
        # 래퍼를 모두 벗겨서 실제 환경 인스턴스에 접근
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
            
        # 환경이 action mask 계산 함수를 제공하는 경우에만 필터링 수행
        if hasattr(base_env, '_compute_action_mask') and hasattr(base_env, 'board'):
            mask = base_env._compute_action_mask(base_env.board)
            
            # 1) 원래 action이 유효하면 그대로 통과
            if mask[action]:
                return action
                
            # 2) 무효이면, 가능한 유효 action 중 첫 번째를 선택(혹은 랜덤 가능)
            valid_actions = np.where(mask)[0]
            if len(valid_actions) > 0:
                # 여기서는 결정적 재현성을 위해 첫 인덱스를 사용
                return valid_actions[0]
            else:
                # 3) 유효 action이 전혀 없다면(이론상 거의 없음) 안전하게 0 반환
                return 0
        else:
            # mask가 없으면 원래 action 그대로 사용
            return action

# -----------------------------------------------------------------------------
# Env factory: 벡터라이즈를 위한 환경 생성기
# -----------------------------------------------------------------------------

def make_env(seed, rows=17, cols=10):
    """FruitBoxEnv 한 개를 만들고 래퍼(필터/정규화/채널/모니터)를 적용하여 반환.
    DummyVecEnv에 넘길 수 있도록 클로저(_thunk)로 감싼다.
    """
    def _thunk():
        base = FruitBoxEnv(FruitBoxConfig(rows=rows, cols=cols, render_mode=None))
        env = ActionFilter(base)              # Invalid action을 valid action으로 변환
        env = ToFloat01(env)                 # 보드 값을 0~1 범위 float32로 정규화
        env = AddChannel(env)                # 관측을 (H, W, 1)로 변경
        env = Monitor(env)                   # episode return/length 로깅
        return env
    return _thunk

# -----------------------------------------------------------------------------
# Main: PPO 학습 파이프라인 구성 및 실행
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    n_envs = 8                              # 병렬 환경 수(샘플 수집 가속)
    seeds = np.arange(n_envs)               # 각 환경에 서로 다른 seed 부여

    # SubprocVecEnv는 마스킹/래퍼 구조에서 이슈가 있을 수 있어 DummyVecEnv 사용
    vec = DummyVecEnv([make_env(int(s)) for s in seeds])

    # 모델 구성 (CnnPolicy가 (C,H,W) 입력을 자동 처리)
    model = PPO(
        policy="CnnPolicy",
        env=vec,
        policy_kwargs=dict(
            features_extractor_class=SmallGridCNN,        # 커스텀 CNN 백본 사용
            features_extractor_kwargs=dict(features_dim=128)  # 특징 벡터 크기
        ),
        # -------- PPO/학습 하이퍼파라미터 --------
        n_steps=2048,                      # rollout 길이(환경당)
        batch_size=1024,                   # 미니배치 크기
        learning_rate=3e-4,                # 학습률
        gamma=0.995,                       # 할인율(장기 보상 강조)
        ent_coef=0.01,                     # 엔트로피 보너스(탐색 유도)
        clip_range=0.2,                    # PPO 클립 범위
        vf_coef=0.5,                       # 가치함수 손실 계수
        n_epochs=4,                        # 각 업데이트 시 미니배치 반복 횟수
        tensorboard_log="runs/fruitbox",   # 텐서보드 로그 경로
        verbose=1,
        seed=42,
    )

    # 평가 환경 (단일 프로세스): 학습 중 주기적으로 정책 성능을 점검
    eval_env = DummyVecEnv([make_env(seed=999)])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="ckpts/",     # 최고 성능 모델 저장 경로
        log_path="eval_logs/",             # 평가 로그 경로
        eval_freq=50_000 // n_envs,         # 평가 주기(전체 스텝 기준)
        n_eval_episodes=10,                 # 평가 에피소드 수
        deterministic=True,                 # 결정적으로 평가(탐색 제외)
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=100_000 // n_envs,        # 체크포인트 저장 주기
        save_path="ckpts/",
        name_prefix="ppo_mask",
    )

    # 학습 시작: 콜백으로 주기적 평가/저장 수행
    model.learn(total_timesteps=1_000_000, callback=[eval_cb, ckpt_cb])

    # 최종 모델 저장
    model.save("fruitbox_ppo_cnn")