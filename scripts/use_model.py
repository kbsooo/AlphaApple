import argparse
import numpy as np

from envs.fruitbox_env import FruitBoxEnv, FruitBoxConfig

# Reuse the same wrappers used during training for consistent preprocessing
from train.train_maskable_ppo import ToFloat01, AddChannel, ActionFilter


def make_eval_env(rows: int, cols: int, use_action_filter: bool = True):
    base = FruitBoxEnv(FruitBoxConfig(rows=rows, cols=cols, render_mode=None))
    env = base
    if use_action_filter:
        env = ActionFilter(env)  # convert invalid actions to valid ones (as in training)
    env = ToFloat01(env)        # scale [0..9] -> [0..1]
    env = AddChannel(env)       # (H, W) -> (H, W, 1)
    return env


def load_model(model_path: str):
    """Try to load as PPO first; if that fails, try MaskablePPO."""
    # Standard PPO
    try:
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    except Exception as e_ppo:
        # Try MaskablePPO
        try:
            from sb3_contrib.ppo_mask import MaskablePPO
            return MaskablePPO.load(model_path)
        except Exception as e_mask:
            raise RuntimeError(
                f"Failed to load model as PPO ({e_ppo}) and as MaskablePPO ({e_mask})."
            )


def play_episode(model, env, deterministic: bool = True):
    obs, info = env.reset()
    done = False
    total = 0.0
    illegal = 0
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, terminated, truncated, info = env.step(int(action))
        total += float(r)
        steps += 1
        if isinstance(info, dict) and info.get("illegal_action", False):
            illegal += 1
        done = terminated or truncated
    return total, illegal, steps


def main():
    ap = argparse.ArgumentParser(description="Run evaluation using a trained FruitBox PPO model.")
    ap.add_argument("--model", default="fruitbox_ppo_cnn.zip", help="Path to saved SB3 model .zip")
    ap.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    ap.add_argument("--rows", type=int, default=17, help="Board rows (must match training)")
    ap.add_argument("--cols", type=int, default=10, help="Board cols (must match training)")
    ap.add_argument("--stochastic", action="store_true", help="Use stochastic policy (default: deterministic)")
    ap.add_argument("--no-filter", action="store_true", help="Disable ActionFilter wrapper")
    args = ap.parse_args()

    model = load_model(args.model)
    env = make_eval_env(args.rows, args.cols, use_action_filter=not args.no_filter)

    # If model is a MaskablePPO, wrap env with ActionMasker to supply action masks
    try:
        from sb3_contrib.ppo_mask import MaskablePPO  # type: ignore
        if isinstance(model, MaskablePPO):
            from sb3_contrib.common.wrappers import ActionMasker  # type: ignore

            def action_masks_fn(e):
                base = e
                while hasattr(base, "env"):
                    base = base.env
                return base._compute_action_mask(base.board)

            env = ActionMasker(env, action_masks_fn)
    except Exception:
        # sb3_contrib may not be installed or model is not MaskablePPO
        pass

    scores = []
    illegals = 0
    steps_sum = 0
    for _ in range(args.episodes):
        s, ill, st = play_episode(model, env, deterministic=not args.stochastic)
        scores.append(s)
        illegals += ill
        steps_sum += st

    print("==== Evaluation ====")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Rows x Cols: {args.rows} x {args.cols}")
    print(f"Deterministic: {not args.stochastic}")
    print(f"Mean score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Mean steps: {steps_sum/args.episodes:.2f}")
    print(f"Illegal actions observed: {illegals}")


if __name__ == "__main__":
    main()
