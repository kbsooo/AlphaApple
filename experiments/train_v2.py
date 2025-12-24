#%% [markdown]
# # FruitBox RL v2 - Dueling Double DQN Training
#
# ### [SOTA Alert]
# - Dueling DQN + Double DQN + PER
# - Q-value clipping으로 폭발 방지
# - 균등 분포 보드 생성으로 일반화 향상
#
# Colab에서 실행 시 GPU 가속 자동 활성화

#%% [code]
import os
import sys
import subprocess
from dataclasses import dataclass

# Colab 환경 감지
IN_COLAB = "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ

# Colab에서 repo 클론
REPO_URL = "https://github.com/kbsooo/AlphaApple.git"
REPO_DIR = "/content/AlphaApple"

if IN_COLAB:
    if not os.path.isdir(REPO_DIR):
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)

    # 필요한 패키지 설치
    try:
        import gymnasium
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium", "matplotlib", "tqdm"])

#%% [code]
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional

# Project root 설정
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

if IN_COLAB:
    project_root = REPO_DIR

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.fruitbox_env_improved import FruitBoxEnvImproved, FruitBoxImprovedConfig
from src.models_v2 import DuelingDQN, get_device
from src.agent_v2 import DoubleDQNAgent, AgentConfig
from src.reward_normalizer import RewardNormalizer

#%% [code]
# Google Drive 마운트 (Colab용)
SAVE_TO_DRIVE = True
if IN_COLAB and SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    checkpoint_dir = "/content/drive/MyDrive/AlphaApple/checkpoints_v2"
else:
    checkpoint_dir = os.path.join(project_root, "checkpoints")

os.makedirs(checkpoint_dir, exist_ok=True)
print(f"Checkpoint directory: {checkpoint_dir}")


#%% [code]
@dataclass
class TrainingConfig:
    """학습 설정."""
    # 환경
    rows: int = 10
    cols: int = 17

    # 학습
    total_episodes: int = 10000
    batch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99

    # Double DQN
    tau: float = 0.005

    # Epsilon-greedy
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_frames: int = 100000

    # PER
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 200000
    buffer_size: int = 100000

    # 커리큘럼
    initial_coverage: float = 0.3
    target_coverage: float = 0.95
    curriculum_episodes: int = 5000
    backward_ratio_start: float = 0.8
    backward_ratio_end: float = 0.3

    # 보상
    normalize_rewards: bool = True
    reward_clip: float = 10.0
    reward_future_action_weight: float = 0.1
    reward_area_bonus_weight: float = 0.5
    reward_area_bonus_base: int = 2
    reward_game_cleared_bonus: float = 50.0

    # 로깅
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 500
    eval_episodes: int = 10

    # 시드
    seed: int = 42


#%% [code]
def set_seed(seed: int):
    """재현성을 위한 시드 설정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    agent: DoubleDQNAgent,
    config: TrainingConfig,
    n_episodes: int = 10
) -> dict:
    """에이전트 평가."""
    env_config = FruitBoxImprovedConfig(
        rows=config.rows,
        cols=config.cols,
        use_backward_generator=False,  # Random 보드로 평가
        target_coverage=1.0,
        reward_future_action_weight=0.0,
        reward_area_bonus_weight=0.0,
        reward_game_cleared_bonus=0.0,
    )
    env = FruitBoxEnvImproved(config=env_config)

    rewards = []
    cleared_ratios = []
    steps_list = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=config.seed + 10000 + ep)
        mask = info["action_mask"]
        initial_nonzero = int(np.count_nonzero(obs))

        episode_reward = 0.0
        steps = 0

        while True:
            action = agent.select_action(obs, mask, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]

            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        remaining = int(np.count_nonzero(obs))
        cleared = (initial_nonzero - remaining) / initial_nonzero if initial_nonzero > 0 else 0

        rewards.append(episode_reward)
        cleared_ratios.append(cleared)
        steps_list.append(steps)

    return {
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "cleared_mean": np.mean(cleared_ratios),
        "cleared_std": np.std(cleared_ratios),
        "steps_mean": np.mean(steps_list),
    }


#%% [code]
def train(config: TrainingConfig):
    """메인 학습 루프."""
    set_seed(config.seed)

    # 환경 설정
    env_config = FruitBoxImprovedConfig(
        rows=config.rows,
        cols=config.cols,
        use_backward_generator=True,
        backward_generator_ratio=config.backward_ratio_start,
        target_coverage=config.initial_coverage,
        reward_future_action_weight=config.reward_future_action_weight,
        reward_area_bonus_weight=config.reward_area_bonus_weight,
        reward_area_bonus_base=config.reward_area_bonus_base,
        reward_game_cleared_bonus=config.reward_game_cleared_bonus,
    )
    env = FruitBoxEnvImproved(config=env_config)

    # 에이전트 설정
    agent_config = AgentConfig(
        rows=config.rows,
        cols=config.cols,
        n_actions=env.n_actions,
        lr=config.learning_rate,
        gamma=config.gamma,
        tau=config.tau,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_frames=config.epsilon_decay_frames,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        per_alpha=config.per_alpha,
        per_beta_start=config.per_beta_start,
        per_beta_frames=config.per_beta_frames,
        normalize_rewards=config.normalize_rewards,
        reward_clip=config.reward_clip,
    )
    agent = DoubleDQNAgent(agent_config)
    print(f"Device: {agent.device}")
    print(f"Model parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")

    # CUDA 최적화
    if agent.device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # 로깅
    episode_rewards = []
    losses = []
    q_values = []
    eval_results = []

    # 학습 루프
    pbar = tqdm(range(config.total_episodes), desc="Training")

    for episode in pbar:
        # 커리큘럼 조정
        progress = min(1.0, episode / config.curriculum_episodes)
        env.cfg.target_coverage = (
            config.initial_coverage +
            progress * (config.target_coverage - config.initial_coverage)
        )
        env.cfg.backward_generator_ratio = (
            config.backward_ratio_start -
            progress * (config.backward_ratio_start - config.backward_ratio_end)
        )

        # 에피소드 시작
        obs, info = env.reset(seed=config.seed + episode)
        mask = info["action_mask"]

        episode_reward = 0.0
        episode_loss = []
        episode_q = []

        while True:
            # Action 선택
            action = agent.select_action(obs, mask, training=True)

            # Step
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            next_mask = next_info["action_mask"]

            # Transition 저장
            agent.push_transition(obs, action, reward, next_obs, done, mask, next_mask)

            # 학습
            result = agent.update()
            if result is not None:
                loss, td_errors, (q_mean, target_mean) = result
                episode_loss.append(loss)
                episode_q.append(q_mean)

            obs = next_obs
            mask = next_mask
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        if episode_q:
            q_values.append(np.mean(episode_q))

        # 로깅
        if episode % config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            avg_q = np.mean(q_values[-100:]) if q_values else 0

            pbar.set_description(
                f"R:{avg_reward:.1f} L:{avg_loss:.4f} Q:{avg_q:.1f} ε:{agent.epsilon:.3f}"
            )

        # 평가
        if episode > 0 and episode % config.eval_interval == 0:
            eval_result = evaluate(agent, config, n_episodes=config.eval_episodes)
            eval_results.append({
                "episode": episode,
                **eval_result
            })
            print(f"\n[Eval] Ep {episode}: "
                  f"R={eval_result['reward_mean']:.1f}±{eval_result['reward_std']:.1f}, "
                  f"Cleared={eval_result['cleared_mean']*100:.1f}%")

        # 체크포인트 저장
        if episode > 0 and episode % config.save_interval == 0:
            save_path = os.path.join(checkpoint_dir, f"dqn_v2_ep{episode}.pt")
            agent.save(save_path)
            print(f"\nSaved: {save_path}")

    # 최종 저장
    final_path = os.path.join(checkpoint_dir, "dqn_v2_final.pt")
    agent.save(final_path)
    print(f"Saved final: {final_path}")

    return agent, {
        "episode_rewards": episode_rewards,
        "losses": losses,
        "q_values": q_values,
        "eval_results": eval_results,
    }


#%% [code]
def plot_results(results: dict, save_path: Optional[str] = None):
    """학습 결과 시각화."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode Rewards
    ax = axes[0, 0]
    rewards = results["episode_rewards"]
    ax.plot(rewards, alpha=0.3, label="Raw")
    # Moving average
    window = 100
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), ma, label=f"MA({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Rewards")
    ax.legend()

    # Training Loss
    ax = axes[0, 1]
    if results["losses"]:
        ax.plot(results["losses"], alpha=0.5)
        if len(results["losses"]) >= window:
            ma = np.convolve(results["losses"], np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(results["losses"])), ma)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")

    # Q-values
    ax = axes[1, 0]
    if results["q_values"]:
        ax.plot(results["q_values"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Q-value")
    ax.set_title("Average Q-values")
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax.axhline(y=200, color='r', linestyle='--', alpha=0.3, label="Max expected")
    ax.legend()

    # Evaluation Cleared Ratio
    ax = axes[1, 1]
    if results["eval_results"]:
        episodes = [r["episode"] for r in results["eval_results"]]
        cleared = [r["cleared_mean"] * 100 for r in results["eval_results"]]
        cleared_std = [r["cleared_std"] * 100 for r in results["eval_results"]]
        ax.errorbar(episodes, cleared, yerr=cleared_std, marker='o', capsize=3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cleared (%)")
    ax.set_title("Evaluation: Cleared Ratio")
    ax.axhline(y=60, color='r', linestyle='--', alpha=0.3, label="Baseline (~60%)")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved: {save_path}")

    plt.show()


#%% [code]
if __name__ == "__main__":
    config = TrainingConfig(
        total_episodes=10000,
        save_interval=1000,
        eval_interval=500,
    )

    print("=" * 60)
    print("FruitBox RL v2 - Dueling Double DQN Training")
    print("=" * 60)
    print(f"Episodes: {config.total_episodes}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Buffer size: {config.buffer_size}")
    print(f"Curriculum: coverage {config.initial_coverage} -> {config.target_coverage}")
    print(f"Backward ratio: {config.backward_ratio_start} -> {config.backward_ratio_end}")
    print("=" * 60)

    agent, results = train(config)

    # 결과 시각화
    plot_path = os.path.join(checkpoint_dir, "training_v2.png")
    plot_results(results, save_path=plot_path)

    # 최종 평가
    print("\n" + "=" * 60)
    print("Final Evaluation (100 episodes on random boards)")
    print("=" * 60)
    final_eval = evaluate(agent, config, n_episodes=100)
    print(f"Reward: {final_eval['reward_mean']:.1f} ± {final_eval['reward_std']:.1f}")
    print(f"Cleared: {final_eval['cleared_mean']*100:.1f}% ± {final_eval['cleared_std']*100:.1f}%")
    print(f"Steps: {final_eval['steps_mean']:.1f}")
