"""
Improved FruitBox environment that addresses several issues in the baseline:
- Optional backward board generation for solvable boards (high coverage).
- Illegal actions advance time and can carry a penalty; episodes end when no legal actions.
- Incremental action-mask updates so we do not rescan every rectangle on illegal steps.
- Reward can include zero-valued cells to encourage 0 활용 전략.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.backward_generator import BackwardBoardGenerator


@dataclass
class FruitBoxImprovedConfig:
    rows: int = 10
    cols: int = 17
    reward_per_cell: float = 1.0
    reward_per_zero_cell: float = 0.0  # zero-valued cells (cleared apples) give no extra reward
    illegal_action_reward: float = -1.0
    max_steps: int = 500  # safety cap; original game uses time, not steps

    # Board generation
    use_backward_generator: bool = True
    target_coverage: float = 0.95  # only used when use_backward_generator is True
    enforce_total_sum_mod_10: bool = True  # fallback random generation

    # Rendering
    render_mode: Optional[str] = None  # "ansi" or None


class FruitBoxEnvImproved(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 30}

    def __init__(self, config: Optional[FruitBoxImprovedConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            cfg = FruitBoxImprovedConfig(**kwargs) if kwargs else FruitBoxImprovedConfig()
        else:
            cfg = config
            for k, v in kwargs.items():
                setattr(cfg, k, v)
        self.cfg: FruitBoxImprovedConfig = cfg

        R, C = self.cfg.rows, self.cfg.cols
        assert R > 0 and C > 0, "rows and cols must be positive"

        # Observation: integers 0..9 (0 means empty)
        self.observation_space = spaces.Box(low=0, high=9, shape=(R, C), dtype=np.int8)

        # Actions: choose any axis-aligned rectangle (r1,c1,r2,c2) with r1<=r2, c1<=c2
        rects = []
        for r1 in range(R):
            for r2 in range(r1, R):
                for c1 in range(C):
                    for c2 in range(c1, C):
                        rects.append((r1, c1, r2, c2))
        self.rects: np.ndarray = np.array(rects, dtype=np.int32)  # (N, 4)
        self.n_actions: int = self.rects.shape[0]
        self.action_space = spaces.Discrete(self.n_actions)

        # Precompute indices for vectorized prefix-sum rectangle queries
        self._idx_r1 = self.rects[:, 0]
        self._idx_c1 = self.rects[:, 1]
        self._idx_r2p = self.rects[:, 2] + 1  # r2+1
        self._idx_c2p = self.rects[:, 3] + 1  # c2+1

        # Cell -> list of rectangles that include the cell (for incremental updates)
        self._cell_to_rects: List[np.ndarray] = self._build_cell_to_rects()

        self.board: np.ndarray = np.zeros((R, C), dtype=np.int16)
        self.steps: int = 0
        self.np_random = np.random.default_rng()

        # Cached per-rect sums and mask
        self._rect_sums: np.ndarray = np.zeros(self.n_actions, dtype=np.int32)
        self._action_mask: np.ndarray = np.zeros(self.n_actions, dtype=bool)

    # ---------- utilities ----------
    def _build_cell_to_rects(self) -> List[np.ndarray]:
        R, C = self.cfg.rows, self.cfg.cols
        mapping: List[List[int]] = [[] for _ in range(R * C)]
        for idx, (r1, c1, r2, c2) in enumerate(self.rects):
            for r in range(r1, r2 + 1):
                base = r * C
                for c in range(c1, c2 + 1):
                    mapping[base + c].append(idx)
        return [np.array(indices, dtype=np.int32) for indices in mapping]

    @staticmethod
    def _padded_prefix_sums(arr: np.ndarray) -> np.ndarray:
        """Return (R+1, C+1) padded summed-area table."""
        R, C = arr.shape
        ps = np.zeros((R + 1, C + 1), dtype=np.int32)
        ps[1:, 1:] = arr.cumsum(axis=0).cumsum(axis=1)
        return ps

    def _rect_sums_vectorized(self, ps: np.ndarray) -> np.ndarray:
        """Compute sums for all rectangles using padded prefix sums (vectorized)."""
        return (
            ps[self._idx_r2p, self._idx_c2p]
            - ps[self._idx_r1, self._idx_c2p]
            - ps[self._idx_r2p, self._idx_c1]
            + ps[self._idx_r1, self._idx_c1]
        )

    def _gen_board(self) -> np.ndarray:
        """Generate a board; prefers solvable boards via backward generator."""
        R, C = self.cfg.rows, self.cfg.cols
        if self.cfg.use_backward_generator:
            gen_seed = int(self.np_random.integers(0, 1_000_000_000))
            generator = BackwardBoardGenerator(rows=R, cols=C, seed=gen_seed)
            board, solution = generator.generate(target_coverage=self.cfg.target_coverage)
            self._last_solution = solution
            return board.astype(np.int16, copy=False)

        # Fallback: random board with sum%10 adjusted
        low, high = 1, 9
        board = self.np_random.integers(low, high + 1, size=(R, C), dtype=np.int16)
        if self.cfg.enforce_total_sum_mod_10:
            delta = int((10 - (board.sum() % 10)) % 10)
            tries = 0
            while delta > 0 and tries < 100:
                r = int(self.np_random.integers(0, R))
                c = int(self.np_random.integers(0, C))
                inc = min(9 - int(board[r, c]), delta)
                if inc > 0:
                    board[r, c] += inc
                    delta -= inc
                tries += 1
        return board

    def _compute_full_mask(self, board: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sums and mask for all rectangles."""
        ps_val = self._padded_prefix_sums(board)
        sums = self._rect_sums_vectorized(ps_val)
        mask = (sums == 10)
        return sums.astype(np.int32, copy=False), mask

    def _update_after_clear(self, r1: int, c1: int, r2: int, c2: int, cleared_vals: np.ndarray):
        """
        Incrementally update rectangle sums/mask after setting a region to zero.
        cleared_vals is the pre-zeroing values of shape (r2-r1+1, c2-c1+1).
        """
        R, C = self.cfg.rows, self.cfg.cols
        deltas: Dict[int, int] = {}
        for dr, row in enumerate(range(r1, r2 + 1)):
            base = row * C
            for dc, col in enumerate(range(c1, c2 + 1)):
                val = int(cleared_vals[dr, dc])
                if val == 0:
                    continue
                cell_rects = self._cell_to_rects[base + col]
                for rect_idx in cell_rects:
                    deltas[rect_idx] = deltas.get(rect_idx, 0) + val

        for rect_idx, delta in deltas.items():
            self._rect_sums[rect_idx] -= delta
            self._action_mask[rect_idx] = (self._rect_sums[rect_idx] == 10)

    # ---------- Gymnasium API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.steps = 0
        self.board = self._gen_board().astype(np.int16, copy=False)
        self._rect_sums, self._action_mask = self._compute_full_mask(self.board)
        info = {"action_mask": self._action_mask}
        obs = self.board.clip(0, 9).astype(np.int8, copy=False)
        return obs, info

    def step(self, action: int):
        assert isinstance(action, (int, np.integer)), "action must be an integer index"
        terminated = False
        truncated = False
        reward = 0.0

        # Illegal action: advance time, optional penalty, end if no legal actions remain.
        if action < 0 or action >= self.n_actions or not self._action_mask[action]:
            self.steps += 1
            reward = float(self.cfg.illegal_action_reward)
            if not self._action_mask.any():
                terminated = True
            if self.steps >= self.cfg.max_steps:
                truncated = True
            obs = self.board.clip(0, 9).astype(np.int8, copy=False)
            info = {"action_mask": self._action_mask, "illegal_action": True}
            return obs, reward, terminated, truncated, info

        r1, c1, r2, c2 = self.rects[action]
        region = self.board[r1 : r2 + 1, c1 : c2 + 1]
        cleared_vals = region.copy()
        cells_total = region.size
        cells_nonzero = int(np.sum(region > 0))
        cells_zero = cells_total - cells_nonzero

        # Apply action
        self.board[r1 : r2 + 1, c1 : c2 + 1] = 0
        self.steps += 1

        reward = (
            self.cfg.reward_per_cell * float(cells_nonzero)
            + self.cfg.reward_per_zero_cell * float(cells_zero)
        )

        # Incremental mask update
        self._update_after_clear(r1, c1, r2, c2, cleared_vals)

        if not self._action_mask.any():
            terminated = True
        if self.steps >= self.cfg.max_steps:
            truncated = True

        obs = self.board.clip(0, 9).astype(np.int8, copy=False)
        info = {"action_mask": self._action_mask, "illegal_action": False}
        return obs, float(reward), terminated, truncated, info

    # ---------- helpers ----------
    def legal_actions(self) -> np.ndarray:
        return np.nonzero(self._action_mask)[0]

    def sample_valid_action(self) -> Optional[int]:
        legal = self.legal_actions()
        if legal.size == 0:
            return None
        return int(self.np_random.choice(legal))

    # ---------- rendering ----------
    def render(self):
        if self.cfg.render_mode != "ansi":
            return
        lines = []
        lines.append(f"Steps={self.steps}")
        lines.append("+" + "---" * self.cfg.cols + "+")
        for r in range(self.cfg.rows):
            row_vals = " ".join(f"{int(v):1d}" for v in self.board[r])
            lines.append(f"| {row_vals} |")
        lines.append("+" + "---" * self.cfg.cols + "+")
        return "\n".join(lines)

    def close(self):
        pass


# ---- quick smoke test ----
if __name__ == "__main__":
    env = FruitBoxEnvImproved(FruitBoxImprovedConfig(render_mode="ansi"))
    obs, info = env.reset(seed=0)
    print("Initial legal actions:", len(np.nonzero(info["action_mask"])[0]))
    total = 0.0
    while True:
        mask = info["action_mask"]
        if not mask.any():
            break
        a = int(np.flatnonzero(mask)[0])
        obs, r, terminated, truncated, info = env.step(a)
        total += r
        if env.cfg.render_mode == "ansi":
            print(env.render())
        if terminated or truncated:
            break
    print("Episode total reward:", total)
