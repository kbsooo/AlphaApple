import numpy as np
from envs.fruitbox_env import FruitBoxEnv, FruitBoxConfig

env = FruitBoxEnv()
total_rewards = []
for _ in range(100):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = env.sample_valid_action()
        if action is None: # No more valid moves
            break
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
    total_rewards.append(episode_reward)

print(f"Random Agent Average Score: {np.mean(total_rewards)}")