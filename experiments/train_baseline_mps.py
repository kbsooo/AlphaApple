# %% [markdown]
# FruitBox RL Baseline Training (DQN) - MPS optimized
#
# This mirrors train_baseline.py but keeps preprocessing, action selection,
# and training on MPS when available.

# %% [code]
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Project root
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from envs.fruitbox_env import FruitBoxEnvImproved, FruitBoxImprovedConfig
from src.agent import ReplayBuffer
from src.models import FruitBoxDQN

# %% [code]
def get_mps_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available. Use train_baseline.py or install a MPS-enabled PyTorch build."
        )
    return torch.device("mps")


class DQNAgentMPS:
    def __init__(
        self,
        rows: int,
        cols: int,
        n_actions: int,
        device: torch.device,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 20000,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update: int = 1000,
    ):
        self.rows = rows
        self.cols = cols
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device

        self.policy_net = FruitBoxDQN(rows, cols, n_actions).to(self.device)
        self.target_net = FruitBoxDQN(rows, cols, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(buffer_size)
        self.steps_done = 0

    def _one_hot_batch(self, states: np.ndarray) -> torch.Tensor:
        state_t = torch.as_tensor(states, device=self.device, dtype=torch.long)
        if state_t.dim() == 2:
            state_t = state_t.unsqueeze(0)
        batch, rows, cols = state_t.shape
        one_hot = torch.zeros(
            (batch, 10, rows, cols), device=self.device, dtype=torch.float32
        )
        one_hot.scatter_(1, state_t.unsqueeze(1), 1.0)
        return one_hot

    def select_action(self, state: np.ndarray, mask: np.ndarray, training: bool = True) -> int:
        self.steps_done += 1

        if training:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
            )

        mask_t = torch.as_tensor(mask, device=self.device, dtype=torch.bool)

        if training and random.random() < self.epsilon:
            legal_idx = torch.nonzero(mask_t, as_tuple=False).squeeze(1)
            if legal_idx.numel() == 0:
                return 0
            choice = legal_idx[
                torch.randint(0, legal_idx.numel(), (1,), device=self.device)
            ]
            return int(choice.item())

        with torch.inference_mode():
            state_batch = self._one_hot_batch(state)
            q_values = self.policy_net(state_batch).squeeze(0)
            masked_q = q_values.masked_fill(~mask_t, -1e9)
            return int(torch.argmax(masked_q).item())

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, masks, next_masks = self.memory.sample(
            self.batch_size
        )

        state_batch = self._one_hot_batch(states)
        next_state_batch = self._one_hot_batch(next_states)
        action_batch = torch.as_tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        done_batch = torch.as_tensor(dones, device=self.device, dtype=torch.bool)
        next_mask_batch = torch.as_tensor(next_masks, device=self.device, dtype=torch.bool)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            next_q_values = next_q_values.masked_fill(~next_mask_batch, -1e9)
            next_state_values = next_q_values.max(1)[0]
            next_state_values = next_state_values.masked_fill(done_batch, 0.0)

        expected_state_action_values = reward_batch + (next_state_values * self.gamma)
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

# %% [code]
# Config
EPISODES = 5000
CURRICULUM_GAP = 500
INITIAL_COVERAGE = 0.3
TARGET_COVERAGE = 0.95
SAVE_INTERVAL = 1000

device = get_mps_device()
print(f"Using device: {device}")

config = FruitBoxImprovedConfig(
    rows=10,
    cols=17,
    use_backward_generator=True,
    target_coverage=INITIAL_COVERAGE,
)
env = FruitBoxEnvImproved(config=config)

agent = DQNAgentMPS(
    rows=env.cfg.rows,
    cols=env.cfg.cols,
    n_actions=env.n_actions,
    device=device,
    batch_size=64,
    epsilon_decay=30000,
)

episode_rewards = []
losses = []
coverages = []

# %% [code]
# Training loop
pbar = tqdm(range(EPISODES))
for episode in pbar:
    if episode > 0 and episode % CURRICULUM_GAP == 0:
        new_coverage = min(TARGET_COVERAGE, env.cfg.target_coverage + 0.1)
        env.cfg.target_coverage = new_coverage
        print(f"\nCurriculum Updated: Target Coverage -> {new_coverage:.2f}")

    obs, info = env.reset()
    mask = info["action_mask"].copy()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(obs, mask, training=True)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        next_mask = next_info["action_mask"].copy()

        agent.memory.push(obs, action, reward, next_obs, done, mask, next_mask)

        loss = agent.update()
        if loss is not None:
            losses.append(loss)

        obs = next_obs
        mask = next_mask
        total_reward += reward

    episode_rewards.append(total_reward)
    coverages.append(env.cfg.target_coverage)

    if episode % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        pbar.set_description(
            f"Eps: {episode} | Avg Rew: {avg_reward:.2f} | Eps: {agent.epsilon:.3f}"
        )

    if episode > 0 and episode % SAVE_INTERVAL == 0:
        agent.save(f"checkpoints/dqn_baseline_eps{episode}.pt")

# %% [markdown]
# ## Training curves

# %% [code]
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()

# %%
