# %% [markdown]
# # FruitBox RL - Batch Evaluation (DQN vs Heuristics)
#
# - Evaluates multiple DQN checkpoints and heuristic policies on identical boards.
# - Reports reward, cleared ratio, and steps with basic plots.

# %% [code]
import argparse
import os
import sys
import random
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# Project root
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from envs.fruitbox_env_improved import FruitBoxEnvImproved, FruitBoxImprovedConfig
from src.agent import DQNAgent
from src.models import FruitBoxDQN, get_device
import torch.nn as nn
import torch.nn.functional as F


class FruitBoxDQNLegacy(nn.Module):
    """Legacy model without BatchNorm for loading old checkpoints."""

    def __init__(self, rows: int, cols: int, n_actions: int):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        flat_features = 64 * rows * cols
        self.fc1 = nn.Linear(flat_features, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "x must be (batch, 10, rows, cols)"
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def is_legacy_checkpoint(path: str) -> bool:
    """Check if checkpoint uses legacy model (no BatchNorm)."""
    ckpt = torch.load(path, map_location="cpu")
    return "bn1.weight" not in ckpt.get("policy_net", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", action="append", required=True, help="Checkpoint path (repeatable)")
    parser.add_argument("--model_label", action="append", default=[], help="Label for each model_path")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--gen_mode", choices=["backward", "random", "mixed"], default="random")
    parser.add_argument("--backward_ratio", type=float, default=0.5)
    parser.add_argument("--target_coverage", type=float, default=0.95)
    parser.add_argument("--reward_per_cell", type=float, default=1.0)
    parser.add_argument("--reward_per_zero_cell", type=float, default=0.0)
    parser.add_argument("--future_action_weight", type=float, default=0.1)
    parser.add_argument("--area_bonus_weight", type=float, default=0.5)
    parser.add_argument("--area_bonus_base", type=int, default=2)
    parser.add_argument("--game_cleared_bonus", type=float, default=50.0)
    parser.add_argument("--no_heuristics", action="store_true")
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> FruitBoxEnvImproved:
    if args.gen_mode == "random":
        use_backward = False
        backward_ratio = 0.0
    elif args.gen_mode == "backward":
        use_backward = True
        backward_ratio = 1.0
    else:
        use_backward = True
        backward_ratio = args.backward_ratio

    cfg = FruitBoxImprovedConfig(
        rows=10,
        cols=17,
        reward_per_cell=args.reward_per_cell,
        reward_per_zero_cell=args.reward_per_zero_cell,
        reward_future_action_weight=args.future_action_weight,
        reward_area_bonus_weight=args.area_bonus_weight,
        reward_area_bonus_base=args.area_bonus_base,
        reward_game_cleared_bonus=args.game_cleared_bonus,
        max_steps=args.max_steps,
        use_backward_generator=use_backward,
        backward_generator_ratio=backward_ratio,
        target_coverage=args.target_coverage,
    )
    return FruitBoxEnvImproved(config=cfg)


def compute_nonzero_counts(env: FruitBoxEnvImproved) -> np.ndarray:
    nz = (env.board > 0).astype(np.int32, copy=False)
    ps = env._padded_prefix_sums(nz)
    counts = (
        ps[env._idx_r2p, env._idx_c2p]
        - ps[env._idx_r1, env._idx_c2p]
        - ps[env._idx_r2p, env._idx_c1]
        + ps[env._idx_r1, env._idx_c1]
    )
    return counts.astype(np.int32, copy=False)


def run_episode_from_board(
    env: FruitBoxEnvImproved,
    board: np.ndarray,
    policy_fn: Callable[[FruitBoxEnvImproved, np.ndarray, np.ndarray], int],
    max_steps: int,
) -> Tuple[float, int, int]:
    env.board = board.copy()
    env.steps = 0
    env._rect_sums, env._action_mask = env._compute_full_mask(env.board)

    total_reward = 0.0
    steps = 0
    while steps < max_steps:
        mask = env._action_mask
        if not mask.any():
            break

        obs = env.board.clip(0, 9).astype(np.int8, copy=False)
        assert obs.shape == (env.cfg.rows, env.cfg.cols), "obs shape mismatch"
        assert mask.shape == (env.n_actions,), "mask shape mismatch"

        action = policy_fn(env, obs, mask)
        if action is None:
            break

        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    remaining = int(np.count_nonzero(env.board))
    return total_reward, steps, remaining


def make_q_policy(agent) -> Callable[[FruitBoxEnvImproved, np.ndarray, np.ndarray], int]:
    def _policy(_env: FruitBoxEnvImproved, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return None
        with torch.no_grad():
            state_v = agent._preprocess(obs).unsqueeze(0)
            q_values = agent.policy_net(state_v).cpu().numpy()[0]
        q_values = np.nan_to_num(q_values, nan=-1e15, neginf=-1e15, posinf=1e15)
        masked_q = np.where(mask, q_values, -1e15)
        return int(np.argmax(masked_q))

    return _policy


def make_random_policy(rng: np.random.Generator) -> Callable[[FruitBoxEnvImproved, np.ndarray, np.ndarray], int]:
    def _policy(_env: FruitBoxEnvImproved, _obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return None
        return int(rng.choice(legal))

    return _policy


def make_max_area_policy(areas: np.ndarray) -> Callable[[FruitBoxEnvImproved, np.ndarray, np.ndarray], int]:
    def _policy(_env: FruitBoxEnvImproved, _obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return None
        return int(legal[int(np.argmax(areas[legal]))])

    return _policy


def make_max_reward_policy(areas: np.ndarray) -> Callable[[FruitBoxEnvImproved, np.ndarray, np.ndarray], int]:
    def _policy(env: FruitBoxEnvImproved, _obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return None

        counts = compute_nonzero_counts(env)
        area_bonus = env.cfg.reward_area_bonus_weight * np.maximum(
            0, areas - env.cfg.reward_area_bonus_base
        )
        reward_est = (
            env.cfg.reward_per_cell * counts
            + env.cfg.reward_per_zero_cell * (areas - counts)
            + area_bonus
        )
        return int(legal[int(np.argmax(reward_est[legal]))])

    return _policy


def summarize(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_label and len(args.model_label) != len(args.model_path):
        raise ValueError("model_label count must match model_path count")

    labels = args.model_label
    if not labels:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in args.model_path]

    for path in args.model_path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    env = build_env(args)
    env.cfg.max_steps = args.max_steps

    rects = env.rects
    areas = (rects[:, 2] - rects[:, 0] + 1) * (rects[:, 3] - rects[:, 1] + 1)
    rng = np.random.default_rng(args.seed)

    policies: List[Tuple[str, Callable[[FruitBoxEnvImproved, np.ndarray, np.ndarray], int]]] = []

    # DQN policies
    for path, label in zip(args.model_path, labels):
        device = get_device()
        if is_legacy_checkpoint(path):
            # Legacy model without BatchNorm
            policy_net = FruitBoxDQNLegacy(env.cfg.rows, env.cfg.cols, env.n_actions).to(device)
        else:
            # New model with BatchNorm
            policy_net = FruitBoxDQN(env.cfg.rows, env.cfg.cols, env.n_actions).to(device)
        ckpt = torch.load(path, map_location=device)
        policy_net.load_state_dict(ckpt["policy_net"])
        policy_net.eval()

        # Create a minimal agent-like object for the policy function
        class MinimalAgent:
            pass

        agent = MinimalAgent()
        agent.policy_net = policy_net
        agent.device = device
        agent.rows = env.cfg.rows
        agent.cols = env.cfg.cols
        agent.epsilon = 0.0

        def _preprocess(state: np.ndarray, dev, rows, cols) -> torch.Tensor:
            state_tensor = torch.tensor(state, dtype=torch.long, device=dev)
            one_hot = torch.nn.functional.one_hot(state_tensor, num_classes=10)
            return one_hot.permute(2, 0, 1).float()

        agent._preprocess = lambda s, d=device, r=env.cfg.rows, c=env.cfg.cols: _preprocess(s, d, r, c)

        policies.append((f"q:{label}", make_q_policy(agent)))

    # Heuristic policies
    if not args.no_heuristics:
        policies.extend(
            [
                ("max_reward", make_max_reward_policy(areas)),
                ("max_area", make_max_area_policy(areas)),
                ("random", make_random_policy(rng)),
            ]
        )

    # Accumulators
    results: Dict[str, Dict[str, List[float]]] = {
        name: {"reward": [], "cleared": [], "steps": []} for name, _ in policies
    }

    # Evaluate each policy on the same initial boards to reduce sampling noise.
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        base_board = env.board.copy()
        initial_nonzero = int(np.count_nonzero(base_board))
        assert obs.shape == (env.cfg.rows, env.cfg.cols), "obs shape mismatch"

        for name, policy_fn in policies:
            total_reward, steps, remaining = run_episode_from_board(
                env, base_board, policy_fn, args.max_steps
            )
            cleared_ratio = 0.0
            if initial_nonzero > 0:
                cleared_ratio = float((initial_nonzero - remaining) / initial_nonzero)
            results[name]["reward"].append(total_reward)
            results[name]["cleared"].append(cleared_ratio)
            results[name]["steps"].append(float(steps))

    # Summary table
    print("\n=== Evaluation Summary ===")
    for name in results:
        reward_mean, reward_std = summarize(results[name]["reward"])
        cleared_mean, cleared_std = summarize(results[name]["cleared"])
        steps_mean, steps_std = summarize(results[name]["steps"])
        print(
            f"{name:12s} | reward {reward_mean:7.2f}±{reward_std:6.2f} | "
            f"cleared {cleared_mean*100:6.2f}%±{cleared_std*100:5.2f}% | "
            f"steps {steps_mean:6.2f}±{steps_std:5.2f}"
        )

    # Plots
    names = list(results.keys())
    reward_means = [summarize(results[n]["reward"])[0] for n in names]
    reward_stds = [summarize(results[n]["reward"])[1] for n in names]
    cleared_means = [summarize(results[n]["cleared"])[0] for n in names]
    cleared_stds = [summarize(results[n]["cleared"])[1] for n in names]

    x = np.arange(len(names))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(x, reward_means, color="#4C78A8")
    plt.errorbar(x, reward_means, yerr=reward_stds, fmt="none", ecolor="black", capsize=3)
    plt.xticks(x, names, rotation=25, ha="right")
    plt.title("Average Reward")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.bar(x, np.array(cleared_means) * 100.0, color="#F58518")
    plt.errorbar(
        x,
        np.array(cleared_means) * 100.0,
        yerr=np.array(cleared_stds) * 100.0,
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    plt.xticks(x, names, rotation=25, ha="right")
    plt.title("Cleared Ratio")
    plt.ylabel("Cleared (%)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
