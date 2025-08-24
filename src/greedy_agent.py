import numpy as np
from envs.fruitbox_env import FruitBoxEnv, FruitBoxConfig

def select_greedy_action_most_cells(env: FruitBoxEnv, legal_actions: np.ndarray) -> int:
    """유효한 행동들 중에서 가장 많은 셀을 제거하는 행동의 인덱스를 반환합니다."""
    best_action = -1
    max_cells = -1

    for action in legal_actions:
        r1, c1, r2, c2 = env.rects[action]
        num_cells = (r2 - r1 + 1) * (c2 - c1 + 1)

        if num_cells > max_cells:
            max_cells = num_cells
            best_action = action
            
    return best_action

# --- 평가 루프 ---
env = FruitBoxEnv()
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    legal_actions = env.legal_actions()
    if len(legal_actions) == 0:
        break
        
    # Greedy 전략에 따라 행동 선택
    action = select_greedy_action_most_cells(env, legal_actions)
    
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Greedy (Most Cells) Agent Score: {total_reward}")