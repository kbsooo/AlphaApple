# %% [markdown]
# # FruitBox RL Baseline Training (Integrated for Colab)
# 
# 이 노트북은 모든 의존성(환경, 모델, 에이전트)을 하나의 파일로 통합하여 
# Google Colab에서 별도의 파일 업로드 없이 즉시 학습을 진행할 수 있도록 합니다.
# GPU(CUDA) 및 TPU 가속을 지원합니다.

# %% [code]
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# %% [markdown]
# ## 1. Backward Board Generator
# 모든 보드가 해답을 가질 수 있도록 역순으로 보드를 생성합니다.

# %% [code]
class BackwardBoardGenerator:
    def __init__(self, rows: int = 10, cols: int = 17, seed: Optional[int] = None):
        self.rows = rows
        self.cols = cols
        self.np_random = np.random.default_rng(seed)
        self.rects = []
        for r1 in range(rows):
            for r2 in range(r1, rows):
                for c1 in range(cols):
                    for c2 in range(c1, cols):
                        self.rects.append((r1, c1, r2, c2))
        self.rects = np.array(self.rects)

    def generate(self, target_coverage: float = 0.9) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        board = np.zeros((self.rows, self.cols), dtype=np.int16)
        solution = []
        total_cells = self.rows * self.cols
        target_filled = int(total_cells * target_coverage)
        filled_count, attempts = 0, 0
        while filled_count < target_filled and attempts < 10000:
            attempts += 1
            idx = self.np_random.integers(0, len(self.rects))
            r1, c1, r2, c2 = self.rects[idx]
            region = board[r1:r2+1, c1:c2+1]
            empty_indices = np.where(region == 0)
            n_empty = len(empty_indices[0])
            if n_empty < 2: continue
            k = self.np_random.integers(2, min(n_empty, 10) + 1)
            dividers = sorted(self.np_random.choice(range(1, 10), k-1, replace=False))
            values = []
            prev = 0
            for d in dividers:
                values.append(d - prev)
                prev = d
            values.append(10 - prev)
            perm = self.np_random.permutation(n_empty)
            for i in range(k):
                rr, cc = empty_indices[0][perm[i]], empty_indices[1][perm[i]]
                board[r1 + rr, c1 + cc] = values[i]
            filled_count += k
            solution.append((r1, c1, r2, c2))
        remaining_zeros = np.where(board == 0)
        for r, c in zip(remaining_zeros[0], remaining_zeros[1]):
            board[r, c] = self.np_random.integers(1, 10)
        return board, solution[::-1]

# %% [markdown]
# ## 2. FruitBox Environment
# 강화학습을 위한 고성능 사과 게임 환경입니다.

# %% [code]
@dataclass
class FruitBoxImprovedConfig:
    rows: int = 10
    cols: int = 17
    reward_per_cell: float = 1.0
    reward_per_zero_cell: float = 0.0
    illegal_action_reward: float = -1.0
    max_steps: int = 500
    use_backward_generator: bool = True
    target_coverage: float = 0.95
    enforce_total_sum_mod_10: bool = True
    render_mode: Optional[str] = None

class FruitBoxEnvImproved(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 30}
    def __init__(self, config: Optional[FruitBoxImprovedConfig] = None, **kwargs):
        super().__init__()
        self.cfg = config if config else FruitBoxImprovedConfig(**kwargs)
        R, C = self.cfg.rows, self.cfg.cols
        self.observation_space = spaces.Box(low=0, high=9, shape=(R, C), dtype=np.int8)
        rects = []
        for r1 in range(R):
            for r2 in range(r1, R):
                for c1 in range(C):
                    for c2 in range(c1, C):
                        rects.append((r1, c1, r2, c2))
        self.rects = np.array(rects, dtype=np.int32)
        self.n_actions = self.rects.shape[0]
        self.action_space = spaces.Discrete(self.n_actions)
        self._idx_r1, self._idx_c1 = self.rects[:, 0], self.rects[:, 1]
        self._idx_r2p, self._idx_c2p = self.rects[:, 2] + 1, self.rects[:, 3] + 1
        self._cell_to_rects = self._build_cell_to_rects()
        self.board = np.zeros((R, C), dtype=np.int16)
        self.steps = 0
        self.np_random = np.random.default_rng()
        self._rect_sums = np.zeros(self.n_actions, dtype=np.int32)
        self._action_mask = np.zeros(self.n_actions, dtype=bool)

    def _build_cell_to_rects(self) -> List[np.ndarray]:
        R, C = self.cfg.rows, self.cfg.cols
        mapping = [[] for _ in range(R * C)]
        for idx, (r1, c1, r2, c2) in enumerate(self.rects):
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    mapping[r * C + c].append(idx)
        return [np.array(indices, dtype=np.int32) for indices in mapping]

    def _padded_prefix_sums(self, arr: np.ndarray) -> np.ndarray:
        R, C = arr.shape
        ps = np.zeros((R + 1, C + 1), dtype=np.int32)
        ps[1:, 1:] = arr.cumsum(axis=0).cumsum(axis=1)
        return ps

    def _rect_sums_vectorized(self, ps: np.ndarray) -> np.ndarray:
        return ps[self._idx_r2p, self._idx_c2p] - ps[self._idx_r1, self._idx_c2p] - ps[self._idx_r2p, self._idx_c1] + ps[self._idx_r1, self._idx_c1]

    def _gen_board(self) -> np.ndarray:
        R, C = self.cfg.rows, self.cfg.cols
        if self.cfg.use_backward_generator:
            generator = BackwardBoardGenerator(rows=R, cols=C, seed=int(self.np_random.integers(0, 10**9)))
            board, _ = generator.generate(target_coverage=self.cfg.target_coverage)
            return board.astype(np.int16)
        board = self.np_random.integers(1, 10, size=(R, C), dtype=np.int16)
        return board

    def _compute_full_mask(self, board: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ps_val = self._padded_prefix_sums(board)
        sums = self._rect_sums_vectorized(ps_val)
        mask = (sums == 10)
        return sums.astype(np.int32), mask

    def _update_after_clear(self, r1: int, c1: int, r2: int, c2: int, cleared_vals: np.ndarray):
        C = self.cfg.cols
        deltas = {}
        for dr, row in enumerate(range(r1, r2 + 1)):
            for dc, col in enumerate(range(c1, c2 + 1)):
                val = int(cleared_vals[dr, dc])
                if val == 0: continue
                for rect_idx in self._cell_to_rects[row * C + col]:
                    deltas[rect_idx] = deltas.get(rect_idx, 0) + val
        for rect_idx, delta in deltas.items():
            self._rect_sums[rect_idx] -= delta
            self._action_mask[rect_idx] = (self._rect_sums[rect_idx] == 10)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None: self.np_random = np.random.default_rng(seed)
        self.steps = 0
        self.board = self._gen_board()
        self._rect_sums, self._action_mask = self._compute_full_mask(self.board)
        return self.board.clip(0, 9).astype(np.int8), {"action_mask": self._action_mask}

    def step(self, action: int):
        if not self._action_mask[action]:
            self.steps += 1
            reward = float(self.cfg.illegal_action_reward)
            terminated = not self._action_mask.any()
            truncated = self.steps >= self.cfg.max_steps
            return self.board.clip(0, 9).astype(np.int8), reward, terminated, truncated, {"action_mask": self._action_mask, "illegal_action": True}
        r1, c1, r2, c2 = self.rects[action]
        region = self.board[r1:r2+1, c1:c2+1]
        cleared_vals = region.copy()
        cells_nonzero = int(np.sum(region > 0))
        cells_zero = region.size - cells_nonzero
        self.board[r1:r2+1, c1:c2+1] = 0
        self.steps += 1
        reward = self.cfg.reward_per_cell * float(cells_nonzero) + self.cfg.reward_per_zero_cell * float(cells_zero)
        self._update_after_clear(r1, c1, r2, c2, cleared_vals)
        terminated = not self._action_mask.any()
        truncated = self.steps >= self.cfg.max_steps
        return self.board.clip(0, 9).astype(np.int8), float(reward), terminated, truncated, {"action_mask": self._action_mask, "illegal_action": False}

# %% [markdown]
# ## 3. DQN Model & Agent
# CNN 기반 아키텍처와 DQN 알고리즘입니다.

# %% [code]
class FruitBoxDQN(nn.Module):
    def __init__(self, rows: int, cols: int, n_actions: int):
        super(FruitBoxDQN, self).__init__()
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * rows * cols, 512)
        self.fc2 = nn.Linear(512, n_actions)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def push(self, *args): self.buffer.append(args)
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return [np.array(x) for x in zip(*samples)]
    def __len__(self): return len(self.buffer)

class DQNAgent:
    def __init__(self, rows, cols, n_actions, device):
        self.rows, self.cols, self.n_actions, self.device = rows, cols, n_actions, device
        self.policy_net = FruitBoxDQN(rows, cols, n_actions).to(device)
        self.target_net = FruitBoxDQN(rows, cols, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(50000)
        self.epsilon, self.epsilon_end, self.epsilon_decay = 1.0, 0.05, 30000
        self.batch_size, self.gamma, self.target_update = 64, 0.99, 1000
        self.steps_done = 0

    def _preprocess(self, state):
        tensor = torch.tensor(state, dtype=torch.long, device=self.device)
        return nn.functional.one_hot(tensor, 10).permute(2, 0, 1).float()

    def select_action(self, state, mask, training=True):
        self.steps_done += 1
        if training: self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 - self.epsilon_end)/self.epsilon_decay)
        if training and random.random() < self.epsilon:
            return random.choice(np.where(mask)[0])
        with torch.no_grad():
            q_values = self.policy_net(self._preprocess(state).unsqueeze(0)).cpu().numpy()[0]
            return int(np.argmax(np.where(mask, q_values, -1e9)))

    def update(self):
        if len(self.memory) < self.batch_size: return None
        s, a, r, ns, d, m, nm = self.memory.sample(self.batch_size)
        sb = torch.stack([self._preprocess(st) for st in s])
        nsb = torch.stack([self._preprocess(st) for st in ns])
        ab, rb, db, nmb = torch.tensor(a, device=self.device).unsqueeze(1), torch.tensor(r, dtype=torch.float32, device=self.device), torch.tensor(d, device=self.device), torch.tensor(nm, device=self.device)
        q_v = self.policy_net(sb).gather(1, ab)
        with torch.no_grad():
            nq_v = self.target_net(nsb)
            nq_v[~nmb] = -1e9
            ns_v = nq_v.max(1)[0]
            ns_v[db] = 0.0
        exp_q_v = (ns_v * self.gamma) + rb
        loss = nn.SmoothL1Loss()(q_v, exp_q_v.unsqueeze(1))
        self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0); self.optimizer.step()
        if self.steps_done % self.target_update == 0: self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

# %% [markdown]
# ## 4. Training Loop
# 커리큘럼 학습을 적용한 학습 루프입니다.

# %% [code]
# Device Selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU (CUDA)")
else:
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print("Using TPU")
    except ImportError:
        device = torch.device("cpu")
        print("Using CPU")

# Config
EPISODES = 10000
CURRICULUM_GAP = 1000
INITIAL_COVERAGE = 0.3
TARGET_COVERAGE = 0.95

env = FruitBoxEnvImproved(target_coverage=INITIAL_COVERAGE)
agent = DQNAgent(env.cfg.rows, env.cfg.cols, env.n_actions, device)

rewards, losses = [], []
pbar = tqdm(range(EPISODES))
for eps in pbar:
    if eps > 0 and eps % CURRICULUM_GAP == 0:
        env.cfg.target_coverage = min(TARGET_COVERAGE, env.cfg.target_coverage + 0.1)
    
    obs, info = env.reset()
    mask, total_reward, done = info["action_mask"], 0, False
    while not done:
        action = agent.select_action(obs, mask)
        n_obs, r, term, trunc, n_info = env.step(action)
        done = term or trunc
        n_mask = n_info["action_mask"]
        agent.memory.push(obs, action, r, n_obs, done, mask, n_mask)
        loss = agent.update()
        if loss: losses.append(loss)
        obs, mask, total_reward = n_obs, n_mask, total_reward + r
    
    rewards.append(total_reward)
    if eps % 10 == 0:
        pbar.set_description(f"Rew: {np.mean(rewards[-10:]):.1f} | Eps: {agent.epsilon:.2f}")

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(rewards); plt.title("Rewards")
plt.subplot(1, 2, 2); plt.plot(losses); plt.title("Loss")
plt.show()

# %% [markdown]
# ## 5. Visualization Test
# 학습된 모델이 실제로 어떻게 플레이하는지 시각화합니다.

# %% [code]
def render_board(board):
    """ASCII 렌더링"""
    lines = ["+---" * board.shape[1] + "+"]
    for row in board:
        line = "|"
        for v in row:
            if v == 0:
                line += " . "
            else:
                line += f" {v} "
        line += "|"
        lines.append(line)
    lines.append("+---" * board.shape[1] + "+")
    return "\n".join(lines)

def visualize_play(agent, env, max_steps=50):
    """학습된 에이전트로 한 게임을 플레이하고 시각화"""
    env.cfg.target_coverage = 0.95
    obs, info = env.reset()
    mask = info["action_mask"]
    
    print("=" * 60)
    print("🍎 FruitBox Visualization - Trained DQN Agent")
    print("=" * 60)
    print(f"\n[Initial Board] (170 cells)")
    print(render_board(obs))
    
    total_reward = 0
    step = 0
    done = False
    
    while not done and step < max_steps:
        action = agent.select_action(obs, mask, training=False)
        r1, c1, r2, c2 = env.rects[action]
        
        n_obs, r, term, trunc, n_info = env.step(action)
        done = term or trunc
        
        step += 1
        total_reward += r
        remaining = np.sum(n_obs > 0)
        
        print(f"\n[Step {step}] Action: ({r1},{c1})-({r2},{c2}) | Cleared: {int(r)} apples | Remaining: {remaining} | Total: {total_reward:.0f}")
        print(render_board(n_obs))
        
        obs = n_obs
        mask = n_info["action_mask"]
    
    print("\n" + "=" * 60)
    print(f"🏆 GAME OVER - Total Reward: {total_reward:.0f} / 170")
    print(f"   Cleared: {(total_reward/170)*100:.1f}% of all apples")
    print("=" * 60)

# Run visualization
visualize_play(agent, env)
