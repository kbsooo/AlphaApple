# baseline_eval.py
import numpy as np
from envs.fruitbox_env import FruitBoxEnv, FruitBoxConfig

def play_episode(env, policy):
    obs, info = env.reset(seed=None)
    total = 0.0
    while True:
        mask = info["action_mask"]
        if not mask.any():
            break
        a = policy(env, mask)  # policy: (env, mask) -> action index (int)
        obs, r, terminated, truncated, info = env.step(int(a))
        total += r
        if terminated or truncated:
            break
    return total

def random_policy(env, mask):
    return int(np.random.choice(np.flatnonzero(mask)))

# Greedy: 지울 수 있는 직사각형 중 "면적(칸 수)" 최대
# env.rects: (N, 4) = (r1,c1,r2,c2) 가정
_areas_cache = None
def _areas(env):
    global _areas_cache
    if _areas_cache is None or len(_areas_cache) != env.n_actions:
        r1, c1, r2, c2 = env.rects[:,0], env.rects[:,1], env.rects[:,2], env.rects[:,3]
        _areas_cache = (r2 - r1 + 1) * (c2 - c1 + 1)
    return _areas_cache

def greedy_policy(env, mask):
    areas = _areas(env)
    legal_idx = np.flatnonzero(mask)
    return int(legal_idx[np.argmax(areas[legal_idx])])

if __name__ == "__main__":
    cfg = FruitBoxConfig(rows=17, cols=10, render_mode=None)
    env = FruitBoxEnv(cfg)

    def run(policy, n=100):
        scores = [play_episode(env, policy) for _ in range(n)]
        return np.mean(scores), np.std(scores)

    m_rand, s_rand = run(random_policy, n=100)
    m_greedy, s_greedy = run(greedy_policy, n=100)

    print(f"Random 100ep: mean={m_rand:.2f}±{s_rand:.2f}")
    print(f"Greedy 100ep: mean={m_greedy:.2f}±{s_greedy:.2f}")