# %% [markdown]
# # FruitBox - Play a Fixed Board with a Trained Model
#
# Loads board data from board.py and runs the trained DQN to visualize steps.

# %% [code]
import argparse
import importlib.util
import os
import sys
import random
import numpy as np
import torch

# Project root
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from envs.fruitbox_env import FruitBoxEnvImproved, FruitBoxImprovedConfig
from src.agent import DQNAgent


def load_board(board_path: str) -> np.ndarray:
    if not os.path.exists(board_path):
        raise FileNotFoundError(f"Board file not found: {board_path}")
    spec = importlib.util.spec_from_file_location("board_module", board_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load board module: {board_path}")
    spec.loader.exec_module(module)
    if not hasattr(module, "board"):
        raise ValueError("Board file must define a variable named 'board'")
    board = np.array(module.board, dtype=np.int16)
    if board.shape != (10, 17):
        raise ValueError(f"Board shape must be (10, 17), got {board.shape}")
    return board


def render_board(board: np.ndarray) -> str:
    lines = ["+" + "---" * board.shape[1] + "+"]
    for row in board:
        line = "| " + " ".join("." if v == 0 else str(int(v)) for v in row) + " |"
        lines.append(line)
    lines.append("+" + "---" * board.shape[1] + "+")
    return "\n".join(lines)


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


def select_action_greedy(agent: DQNAgent, obs: np.ndarray, mask: np.ndarray):
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return None
    with torch.no_grad():
        state_v = agent._preprocess(obs).unsqueeze(0)
        q_values = agent.policy_net(state_v).cpu().numpy()[0]
    q_values = np.nan_to_num(q_values, nan=-1e15, neginf=-1e15, posinf=1e15)
    best_idx = legal[int(np.argmax(q_values[legal]))]
    return int(best_idx)


def select_action_policy(
    policy: str,
    agent: DQNAgent,
    env: FruitBoxEnvImproved,
    obs: np.ndarray,
    mask: np.ndarray,
    areas: np.ndarray,
):
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return None

    if policy == "q":
        return select_action_greedy(agent, obs, mask)

    if policy == "max_reward":
        counts = compute_nonzero_counts(env)
        reward_est = (
            env.cfg.reward_per_cell * counts
            + env.cfg.reward_per_zero_cell * (areas - counts)
        )
        return int(legal[int(np.argmax(reward_est[legal]))])

    if policy == "max_area":
        return int(legal[int(np.argmax(areas[legal]))])

    if policy == "random":
        return int(random.choice(legal.tolist()))

    raise ValueError(f"Unknown policy: {policy}")


def run_episode(
    policy: str,
    agent: DQNAgent,
    env: FruitBoxEnvImproved,
    board: np.ndarray,
    max_steps: int,
    render_every: int,
    areas: np.ndarray,
):
    env.board = board.copy()
    env.steps = 0
    env._rect_sums, env._action_mask = env._compute_full_mask(env.board)

    total_reward = 0.0
    step = 0

    print(f"\nPolicy: {policy}")
    print("Initial board:")
    print(render_board(env.board))

    torch.set_grad_enabled(False)
    while step < max_steps:
        mask = env._action_mask
        if not mask.any():
            break

        obs = env.board.clip(0, 9).astype(np.int8, copy=False)
        action = select_action_policy(policy, agent, env, obs, mask, areas)
        if action is None:
            break

        r1, c1, r2, c2 = env.rects[action]
        region_sum = int(env.board[r1 : r2 + 1, c1 : c2 + 1].sum())
        if not mask[action]:
            print(f"Step {step}: illegal action {action} -> ({r1},{c1})-({r2},{c2})")
            break

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        print(
            f"\nStep {step}: action {action} ({r1},{c1})-({r2},{c2}) "
            f"sum={region_sum} reward={reward:.1f} remaining={int(np.sum(obs > 0))}"
        )
        if render_every > 0 and step % render_every == 0:
            print(render_board(obs))

        if terminated or truncated:
            break

    remaining = int(np.sum(env.board > 0))
    print("\nGame finished.")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Steps: {step}")
    print(f"Remaining: {remaining}")
    return total_reward, step, remaining


# %% [code]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/dqn_colab_final.pt")
    parser.add_argument("--board_path", type=str, default="board.py")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--render_every", type=int, default=1)
    parser.add_argument(
        "--policy",
        type=str,
        default="q",
        choices=["q", "max_reward", "max_area", "random"],
        help="Policy for action selection",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run multiple policies for comparison",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    board = load_board(args.board_path)

    config = FruitBoxImprovedConfig(
        rows=10,
        cols=17,
        use_backward_generator=False,
        target_coverage=1.0,
    )
    env = FruitBoxEnvImproved(config=config)

    agent = DQNAgent(
        rows=env.cfg.rows,
        cols=env.cfg.cols,
        n_actions=env.n_actions,
        batch_size=64,
        epsilon_decay=30000,
    )
    agent.load(args.model_path)

    rects = env.rects
    areas = (rects[:, 2] - rects[:, 0] + 1) * (rects[:, 3] - rects[:, 1] + 1)

    policies = [args.policy]
    if args.compare:
        policies = ["q", "max_reward", "max_area", "random"]

    summary = []
    for policy in policies:
        reward, steps, remaining = run_episode(
            policy,
            agent,
            env,
            board,
            args.max_steps,
            args.render_every,
            areas,
        )
        summary.append((policy, reward, steps, remaining))

    if len(summary) > 1:
        print("\nSummary:")
        for policy, reward, steps, remaining in summary:
            print(
                f"- {policy}: reward={reward:.1f}, steps={steps}, remaining={remaining}"
            )


if __name__ == "__main__":
    main()
