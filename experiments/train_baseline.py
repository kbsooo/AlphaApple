# %% [markdown]
# # FruitBox RL Baseline Training (DQN)
# 
# 이 노트북은 DQN을 사용하여 사과 게임 환경을 해결하는 베이스라인 모델을 학습합니다.
# `BackwardBoardGenerator`를 활용한 커리큘럼 학습을 적용합니다.

# %% [code]
import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 추가
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    # Jupyter Notebook 위에서 실행될 경우 __file__ 이 정의되지 않으므로 현재 작업 디렉토리 기준 처리
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from envs.fruitbox_env import FruitBoxEnvImproved, FruitBoxImprovedConfig
from src.agent import DQNAgent

# %% [code]
# 설정
EPISODES = 5000
CURRICULUM_GAP = 500 # n 에피소드마다 커버리지 증가
INITIAL_COVERAGE = 0.3
TARGET_COVERAGE = 0.95
SAVE_INTERVAL = 1000
BACKWARD_RATIO_START = 1.0
BACKWARD_RATIO_MIN = 0.2
BACKWARD_RATIO_DECAY_EPISODES = EPISODES
FUTURE_ACTION_WEIGHT = 0.1
AREA_BONUS_WEIGHT = 0.5
AREA_BONUS_BASE = 2
GAME_CLEARED_BONUS = 50.0

# 환경 초기화
config = FruitBoxImprovedConfig(
    rows=10, 
    cols=17, 
    use_backward_generator=True,
    backward_generator_ratio=BACKWARD_RATIO_START,
    target_coverage=INITIAL_COVERAGE,
    reward_future_action_weight=FUTURE_ACTION_WEIGHT,
    reward_area_bonus_weight=AREA_BONUS_WEIGHT,
    reward_area_bonus_base=AREA_BONUS_BASE,
    reward_game_cleared_bonus=GAME_CLEARED_BONUS,
)
env = FruitBoxEnvImproved(config=config)

# 에이전트 초기화
agent = DQNAgent(
    rows=env.cfg.rows,
    cols=env.cfg.cols,
    n_actions=env.n_actions,
    batch_size=64,
    epsilon_decay=30000 # 탐험을 충분히 하도록 길게 설정
)

# 로그 기록용
episode_rewards = []
losses = []
coverages = []
backward_ratios = []
avg_valid_actions = []

# %% [code]
# 학습 루프
pbar = tqdm(range(EPISODES))
for episode in pbar:
    ratio_progress = min(1.0, episode / max(1, BACKWARD_RATIO_DECAY_EPISODES))
    backward_ratio = max(
        BACKWARD_RATIO_MIN,
        BACKWARD_RATIO_START - ratio_progress * (BACKWARD_RATIO_START - BACKWARD_RATIO_MIN),
    )
    env.cfg.backward_generator_ratio = backward_ratio

    # 커리큘럼 업데이트
    if episode > 0 and episode % CURRICULUM_GAP == 0:
        new_coverage = min(TARGET_COVERAGE, env.cfg.target_coverage + 0.1)
        env.cfg.target_coverage = new_coverage
        print(f"\nCurriculum Updated: Target Coverage -> {new_coverage:.2f}")

    obs, info = env.reset()
    mask = info["action_mask"]
    total_reward = 0
    done = False
    valid_action_counts = []
    
    while not done:
        valid_action_counts.append(int(mask.sum()))
        action = agent.select_action(obs, mask, training=True)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        next_mask = next_info["action_mask"]
        
        # 메모리 저장
        agent.memory.push(obs, action, reward, next_obs, done, mask, next_mask)
        
        # 학습
        loss = agent.update()
        if loss is not None:
            losses.append(loss)
            
        obs = next_obs
        mask = next_mask
        total_reward += reward
        
    episode_rewards.append(total_reward)
    coverages.append(env.cfg.target_coverage)
    backward_ratios.append(env.cfg.backward_generator_ratio)
    avg_valid_actions.append(float(np.mean(valid_action_counts)) if valid_action_counts else 0.0)
    
    # 출력
    if episode % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        pbar.set_description(f"Eps: {episode} | Avg Rew: {avg_reward:.2f} | Eps: {agent.epsilon:.3f}")

    if episode > 0 and episode % SAVE_INTERVAL == 0:
        agent.save(f"checkpoints/dqn_baseline_eps{episode}.pt")

# %% [markdown]
# ## 학습 결과 시각화

# %% [code]
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(episode_rewards)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.subplot(2, 2, 2)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")

plt.subplot(2, 2, 3)
plt.plot(avg_valid_actions)
plt.title("Avg Valid Actions")
plt.xlabel("Episode")
plt.ylabel("Count")

plt.subplot(2, 2, 4)
plt.plot(backward_ratios, label="Backward Ratio")
plt.plot(coverages, label="Target Coverage")
plt.title("Generation Schedule")
plt.xlabel("Episode")
plt.ylabel("Ratio")
plt.legend()
plt.show()
