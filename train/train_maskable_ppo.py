# train_maskable_ppo.py
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

# Custom CNN feature extractor for small grids
class SmallGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # We know our input is (17, 10, 1)
        n_input_channels = observation_space.shape[2]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # downsample
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float().permute(0, 3, 1, 2)
            ).shape[1]
        
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, height, width, channel) -> (batch, channel, height, width)
        observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))

# ---- Obs wrappers ----
class ToFloat01(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(h, w), dtype=np.float32)
    def observation(self, obs):
        return (obs.astype(np.float32) / 9.0)

class AddChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(h, w, 1), dtype=np.float32)
    def observation(self, obs):
        return obs[..., None]  # (H, W, 1)

# ---- Action filtering wrapper ----
class ActionFilter(gym.ActionWrapper):
    """Convert invalid actions to valid ones automatically"""
    def __init__(self, env):
        super().__init__(env)
        
    def action(self, action):
        # Get base environment to access action mask
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
            
        if hasattr(base_env, '_compute_action_mask') and hasattr(base_env, 'board'):
            mask = base_env._compute_action_mask(base_env.board)
            
            # If action is valid, use it
            if mask[action]:
                return action
                
            # If action is invalid, find a valid one
            valid_actions = np.where(mask)[0]
            if len(valid_actions) > 0:
                # Use the first valid action (or could use random choice)
                return valid_actions[0]
            else:
                # Fallback: return action 0 (should not happen in practice)
                return 0
        else:
            # Fallback: pass through original action
            return action

def make_env(seed, rows=17, cols=10):
    def _thunk():
        base = FruitBoxEnv(FruitBoxConfig(rows=rows, cols=cols, render_mode=None))
        env = ActionFilter(base)              # Invalid action을 valid action으로 변환
        env = ToFloat01(env)
        env = AddChannel(env)                 # (H, W, 1)
        env = Monitor(env)                    # episode returns 기록
        return env
    return _thunk

if __name__ == "__main__":
    n_envs = 8
    seeds = np.arange(n_envs)
    # Use DummyVecEnv instead of SubprocVecEnv to avoid multiprocessing issues with action masking
    vec = DummyVecEnv([make_env(int(s)) for s in seeds])

    # 모델 구성 (CnnPolicy가 (C,H,W) 입력을 자동 처리)
    model = PPO(
        policy="CnnPolicy",
        env=vec,
        policy_kwargs=dict(
            features_extractor_class=SmallGridCNN,
            features_extractor_kwargs=dict(features_dim=128)
        ),
        n_steps=2048,
        batch_size=1024,
        learning_rate=3e-4,
        gamma=0.995,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
        n_epochs=4,
        tensorboard_log="runs/fruitbox",
        verbose=1,
        seed=42,
    )

    # 평가 환경 (단일 프로세스)
    eval_env = DummyVecEnv([make_env(seed=999)])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="ckpts/",
        log_path="eval_logs/",
        eval_freq=50_000 // n_envs,   # 스텝 수는 총합 기준
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(save_freq=100_000 // n_envs, save_path="ckpts/", name_prefix="ppo_mask")

    model.learn(total_timesteps=1_000_000, callback=[eval_cb, ckpt_cb])
    model.save("fruitbox_ppo_cnn")