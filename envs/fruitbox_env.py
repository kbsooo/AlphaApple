from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class FruitBoxConfig:
    rows: int = 17
    cols: int = 10
    reward_per_cell: float = 1.0
    max_steps: int = 500  # safety cap; original game uses time, not steps

    # Board generation
    values_low: int = 1
    values_high: int = 9  # inclusive
    enforce_total_sum_mod_10: bool = True  # try to make sum % 10 == 0 at reset

    # Rendering
    render_mode: Optional[str] = None  # "ansi" or None


class FruitBoxEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 30}

    def __init__(self, config: Optional[FruitBoxConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            cfg = FruitBoxConfig(**kwargs) if kwargs else FruitBoxConfig()
        else:
            cfg = config
            for k, v in kwargs.items():
                setattr(cfg, k, v)
        self.cfg: FruitBoxConfig = cfg

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

        self.board: np.ndarray = np.zeros((R, C), dtype=np.int16)
        self.steps: int = 0
        self.np_random = np.random.default_rng()

    # ---------- utilities ----------
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
        R, C = self.cfg.rows, self.cfg.cols
        low, high = self.cfg.values_low, self.cfg.values_high
        assert 1 <= low <= high <= 9, "values must be between 1 and 9 inclusive"

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
            # solvability is NOT guaranteed; we simply adjust total modulo if possible
        return board

    def _compute_action_mask(self, board: np.ndarray) -> np.ndarray:
        """
        Boolean mask of shape (n_actions,):
        True for rectangles whose sum==10 AND contain no zeros.
        """
        ps_val = self._padded_prefix_sums(board)
        sums = self._rect_sums_vectorized(ps_val)

        zero_grid = (board == 0).astype(np.int8)
        ps_zero = self._padded_prefix_sums(zero_grid)
        zero_counts = self._rect_sums_vectorized(ps_zero)

        return (sums == 10) & (zero_counts == 0)

    # ---------- Gymnasium API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.steps = 0
        self.board = self._gen_board().astype(np.int16, copy=False)
        mask = self._compute_action_mask(self.board)
        info = {"action_mask": mask}
        obs = self.board.clip(0, 9).astype(np.int8, copy=False)
        return obs, info

    def step(self, action: int):
        assert isinstance(action, (int, np.integer)), "action must be an integer index"
        terminated = False
        truncated = False
        reward = 0.0

        mask = self._compute_action_mask(self.board)
        if action < 0 or action >= self.n_actions or not mask[action]:
            # Illegal actions are ignored (no state change, no reward), like UI rejection.
            obs = self.board.clip(0, 9).astype(np.int8, copy=False)
            info = {"action_mask": mask, "illegal_action": True}
            return obs, 0.0, False, False, info

        r1, c1, r2, c2 = self.rects[action]
        region_h = (r2 - r1 + 1)
        region_w = (c2 - c1 + 1)
        cells_cleared = region_h * region_w

        self.board[r1 : r2 + 1, c1 : c2 + 1] = 0

        reward = self.cfg.reward_per_cell * float(cells_cleared)
        self.steps += 1

        # Termination: no valid actions remain OR safety cap reached
        new_mask = self._compute_action_mask(self.board)
        if not new_mask.any():
            terminated = True
        if self.steps >= self.cfg.max_steps:
            truncated = True

        obs = self.board.clip(0, 9).astype(np.int8, copy=False)
        info = {"action_mask": new_mask, "illegal_action": False}
        return obs, float(reward), terminated, truncated, info

    # ---------- helpers ----------
    def legal_actions(self) -> np.ndarray:
        return np.nonzero(self._compute_action_mask(self.board))[0]

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
    env = FruitBoxEnv(FruitBoxConfig(render_mode="ansi"))
    obs, info = env.reset(seed=0)
    print("Initial legal actions:", len(np.nonzero(info["action_mask"])[0]))
    done = False
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
