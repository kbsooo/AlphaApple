# %% [markdown]
# # FruitBox RL Baseline Training (Integrated JAX/TPU)
# 
# 이 노트북은 **JAX/Flax**를 사용하여 사과 게임 환경을 TPU에서 고속으로 학습합니다.
# 모든 의존성이 포함되어 있어 Google Colab에서 즉시 실행이 가능합니다.

# %% [code]
# 필요 라이브러리 설치 (Colab 환경용)
# !pip install -q flax optax gymnasium

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# %% [markdown]
# ## 1. Backward Board Generator
# 모든 보드가 해답을 가질 수 있도록 역순으로 보드를 생성합니다. (NumPy 기반)

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
# 강화학습을 위한 고성능 사과 게임 환경입니다. (NumPy 기반)

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
        return self.np_random.integers(1, 10, size=(R, C), dtype=np.int16)

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
# ## 3. JAX/Flax Model & Agent
# Flax를 사용한 신경망 모델과 JAX 기반의 함수형 학습 로직입니다.

# %% [code]
class FruitBoxDQN(nn.Module):
    rows: int
    cols: int
    n_actions: int

    @nn.compact
    def __call__(self, x):
        # x shape: (Batch, 10, rows, cols)
        # JAX Conv expects (Batch, Height, Width, Channels) or manually permuted
        x = x.transpose((0, 2, 3, 1)) # (B, R, C, 10)
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1)) # Flatten
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def push(self, *args): self.buffer.append(args)
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return [np.array(x) for x in zip(*samples)]
    def __len__(self): return len(self.buffer)

def preprocess(state):
    # state: (R, C)
    # One-hot encode to (10, R, C)
    one_hot = jax.nn.one_hot(state, 10) # (R, C, 10)
    return one_hot.transpose((2, 0, 1)) # (10, R, C)

@jax.jit
def train_step(state, batch, gamma):
    s, a, r, ns, d, m, nm = batch
    
    def loss_fn(params):
        # Q(s, a)
        q_values = state.apply_fn({'params': params}, s) # (B, n_actions)
        q_v = jnp.take_along_axis(q_values, a[:, None], axis=1).squeeze()
        
        # max Q(ns, na)
        next_q_values = state.apply_fn({'params': state.params}, ns) # Using same params for now, or target
        # Apply mask
        next_q_values = jnp.where(nm, next_q_values, -1e9)
        max_next_q = jnp.max(next_q_values, axis=1)
        max_next_q = jnp.where(d, 0.0, max_next_q) # Done states have 0 future value
        
        target_q = r + gamma * max_next_q
        loss = jnp.mean((q_v - target_q)**2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# %% [markdown]
# ## 4. Training Loop (Integrated)

# %% [code]
# Device Selection
device = jax.devices()[0]
print(f"Using device: {device}")

# Config
EPISODES = 5000
CURRICULUM_GAP = 500
INITIAL_COVERAGE = 0.3
TARGET_COVERAGE = 0.95
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 30000

env = FruitBoxEnvImproved(target_coverage=INITIAL_COVERAGE)
model = FruitBoxDQN(env.cfg.rows, env.cfg.cols, env.n_actions)

# Initialize state
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
dummy_s = jnp.zeros((1, 10, env.cfg.rows, env.cfg.cols))
params = model.init(init_rng, dummy_s)['params']
tx = optax.adam(learning_rate=1e-4)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

memory = ReplayBuffer(50000)
epsilon = EPSILON_START

# Stats
rewards, losses = [], []
pbar = tqdm(range(EPISODES))

for eps in pbar:
    if eps > 0 and eps % CURRICULUM_GAP == 0:
        env.cfg.target_coverage = min(TARGET_COVERAGE, env.cfg.target_coverage + 0.1)
    
    obs, info = env.reset()
    mask, total_reward, done = info["action_mask"], 0, False
    
    while not done:
        # Action selection
        if random.random() < epsilon:
            action = random.choice(np.where(mask)[0])
        else:
            s_processed = preprocess(obs)[None, ...]
            q_values = model.apply({'params': state.params}, s_processed)
            q_values = np.where(mask, q_values[0], -1e9)
            action = int(np.argmax(q_values))
        
        epsilon = max(EPSILON_END, epsilon - (EPSILON_START - EPSILON_END)/EPSILON_DECAY)
        
        n_obs, r, term, trunc, n_info = env.step(action)
        done = term or trunc
        n_mask = n_info["action_mask"]
        
        # Convert to arrays and push to memory
        memory.push(
            preprocess(obs),
            action,
            r,
            preprocess(n_obs),
            done,
            mask,
            n_mask
        )
        
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            # Convert batch list elements to jnp arrays
            batch_jnp = [jnp.array(x) for x in batch]
            state, loss = train_step(state, batch_jnp, GAMMA)
            losses.append(float(loss))
            
        obs, mask, total_reward = n_obs, n_mask, total_reward + r
    
    rewards.append(total_reward)
    if eps % 10 == 0:
        avg_rew = np.mean(rewards[-10:]) if rewards else 0.0
        pbar.set_description(f"Rew: {avg_rew:.1f} | Eps: {epsilon:.2f}")

# Visualizing results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(rewards); plt.title("Rewards")
plt.subplot(1, 2, 2); plt.plot(losses); plt.title("Loss")
plt.show()
