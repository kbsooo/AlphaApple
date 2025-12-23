# %% [markdown]
# # FruitBox RL - Colab + Google Drive Training
#
# - Top cell can clone the repo into `/content/AlphaApple`.
# - Uses the repo environment (`envs/fruitbox_env_improved.py`) as the single source of truth.
# - Mixed board generation reduces distribution shift.
# - Reward shaping encourages future options and larger clears.
# - Checkpoints save directly to Google Drive when running on Colab.

# %% [markdown]
# ### [SOTA Alert]
# PyTorch 2.x supports `torch.compile` for fused kernels on CUDA. Toggle it below if you want extra speed.

# %% [code]
import os
import sys
import subprocess

# Colab detection
IN_COLAB = "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ

# Repo bootstrap (Colab only)
REPO_URL = "https://github.com/kbsooo/AlphaApple.git"
REPO_DIR = "/content/AlphaApple"
COLAB_PROJECT_ROOT = None

if IN_COLAB:
    if not os.path.isdir(REPO_DIR):
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    else:
        subprocess.check_call(["git", "-C", REPO_DIR, "pull"])
    COLAB_PROJECT_ROOT = REPO_DIR
    os.chdir(REPO_DIR)

if IN_COLAB:
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gymnasium", "matplotlib", "tqdm"]
        )

import random
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Resolve project root
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

if COLAB_PROJECT_ROOT:
    project_root = COLAB_PROJECT_ROOT

if project_root not in sys.path:
    sys.path.append(project_root)

if not os.path.isdir(os.path.join(project_root, "envs")):
    raise FileNotFoundError(
        "Project root not found. Clone the repo to /content/AlphaApple or set project_root."
    )

# Optional Google Drive mount for checkpoints
SAVE_TO_DRIVE = True
if IN_COLAB and SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    drive_root = "/content/drive/MyDrive/AlphaApple"
else:
    drive_root = project_root

checkpoint_dir = os.path.join(drive_root, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"Checkpoint dir: {checkpoint_dir}")

from envs.fruitbox_env_improved import FruitBoxEnvImproved, FruitBoxImprovedConfig
from src.agent import DQNAgent

# %% [code]
# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Config
EPISODES = 5000
CURRICULUM_GAP = 500
INITIAL_COVERAGE = 0.3
TARGET_COVERAGE = 0.95
SAVE_INTERVAL = 1000
BACKWARD_RATIO_START = 1.0
BACKWARD_RATIO_MIN = 0.2
BACKWARD_RATIO_DECAY_EPISODES = EPISODES
FUTURE_ACTION_WEIGHT = 0.1  # more legal actions -> more long-term options
AREA_BONUS_WEIGHT = 0.5  # prefer larger clears to avoid greedy micro-moves
AREA_BONUS_BASE = 2  # bonus starts above this area size
GAME_CLEARED_BONUS = 50.0  # terminal bonus to push full clears
LOG_EVERY = 10

# Optional PyTorch 2.x compile (CUDA only)
USE_TORCH_COMPILE = False

# %% [code]
# Environment
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

# Agent
agent = DQNAgent(
    rows=env.cfg.rows,
    cols=env.cfg.cols,
    n_actions=env.n_actions,
    batch_size=64,
    epsilon_decay=30000,
)
print(f"Using device: {agent.device}")

if agent.device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # fixed input size -> faster convs

if USE_TORCH_COMPILE and agent.device.type == "cuda" and hasattr(torch, "compile"):
    agent.policy_net = torch.compile(agent.policy_net)
    agent.target_net = torch.compile(agent.target_net)

# %% [code]
# Training loop
episode_rewards = []
losses = []
coverages = []
backward_ratios = []
avg_valid_actions = []

pbar = tqdm(range(EPISODES))
for episode in pbar:
    ratio_progress = min(1.0, episode / max(1, BACKWARD_RATIO_DECAY_EPISODES))
    backward_ratio = max(
        BACKWARD_RATIO_MIN,
        BACKWARD_RATIO_START - ratio_progress * (BACKWARD_RATIO_START - BACKWARD_RATIO_MIN),
    )
    env.cfg.backward_generator_ratio = backward_ratio

    if episode > 0 and episode % CURRICULUM_GAP == 0:
        new_coverage = min(TARGET_COVERAGE, env.cfg.target_coverage + 0.1)
        env.cfg.target_coverage = new_coverage
        print(f"\nCurriculum Updated: Target Coverage -> {new_coverage:.2f}")

    obs, info = env.reset(seed=SEED + episode)
    mask = info["action_mask"]
    assert obs.shape == (env.cfg.rows, env.cfg.cols), "obs shape mismatch"
    assert mask.shape == (env.n_actions,), "mask shape mismatch"

    total_reward = 0.0
    done = False
    valid_action_counts = []

    while not done:
        valid_action_counts.append(int(mask.sum()))
        action = agent.select_action(obs, mask, training=True)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        next_mask = next_info["action_mask"]
        assert next_obs.shape == (env.cfg.rows, env.cfg.cols), "next_obs shape mismatch"
        assert next_mask.shape == (env.n_actions,), "next_mask shape mismatch"

        agent.memory.push(obs, action, reward, next_obs, done, mask, next_mask)

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

    if episode % LOG_EVERY == 0:
        avg_reward = np.mean(episode_rewards[-LOG_EVERY:])
        pbar.set_description(f"Eps: {episode} | Avg Rew: {avg_reward:.2f} | Eps: {agent.epsilon:.3f}")

    if episode > 0 and episode % SAVE_INTERVAL == 0:
        save_path = os.path.join(checkpoint_dir, f"dqn_colab_eps{episode}.pt")
        agent.save(save_path)
        print(f"Saved checkpoint: {save_path}")

# Final save
final_path = os.path.join(checkpoint_dir, "dqn_colab_final.pt")
agent.save(final_path)
print(f"Saved final checkpoint: {final_path}")

# %% [markdown]
# ## Training Curves

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

plt.tight_layout()
plt.show()
