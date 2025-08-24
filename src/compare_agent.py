import numpy as np
from tqdm import tqdm  # 진행률 표시를 위한 라이브러리 (pip install tqdm)
from envs.fruitbox_env import FruitBoxEnv

# --------------------------------------------------------------------------
# Greedy Agent 로직 (제공된 코드와 동일)
# --------------------------------------------------------------------------
def select_greedy_action_most_cells(env: FruitBoxEnv, legal_actions: np.ndarray) -> int:
    """
    유효한 행동들 중에서 가장 많은 셀을 제거하는 행동의 인덱스를 반환합니다.
    """
    best_action = -1
    max_cells = -1

    for action in legal_actions:
        r1, c1, r2, c2 = env.rects[action]
        num_cells = (r2 - r1 + 1) * (c2 - c1 + 1)

        if num_cells > max_cells:
            max_cells = num_cells
            best_action = action
            
    return best_action

# --------------------------------------------------------------------------
# 평가 실행 함수들
# --------------------------------------------------------------------------
def evaluate_random_agent(env: FruitBoxEnv, num_episodes: int) -> list[float]:
    """랜덤 에이전트를 지정된 횟수만큼 실행하고 보상 리스트를 반환합니다."""
    total_rewards = []
    print("Evaluating Random Agent...")
    for _ in tqdm(range(num_episodes)):
        env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = env.sample_valid_action()
            if action is None:  # No more valid moves
                break
            _, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
    return total_rewards

def evaluate_greedy_agent(env: FruitBoxEnv, num_episodes: int) -> list[float]:
    """Greedy 에이전트를 지정된 횟수만큼 실행하고 보상 리스트를 반환합니다."""
    total_rewards = []
    print("Evaluating Greedy Agent (Most Cells)...")
    for _ in tqdm(range(num_episodes)):
        env.reset()
        episode_reward = 0
        done = False
        while not done:
            legal_actions = env.legal_actions()
            if len(legal_actions) == 0:
                break
            
            action = select_greedy_action_most_cells(env, legal_actions)
            
            _, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
    return total_rewards

# --------------------------------------------------------------------------
# 메인 실행 블록
# --------------------------------------------------------------------------
if __name__ == "__main__":
    NUM_EPISODES = 100
    
    # 환경은 한 번만 생성하여 공유합니다.
    env = FruitBoxEnv()

    # 각 에이전트 평가 실행
    random_rewards = evaluate_random_agent(env, NUM_EPISODES)
    greedy_rewards = evaluate_greedy_agent(env, NUM_EPISODES)
    
    # 결과 계산 및 출력
    avg_random_score = np.mean(random_rewards)
    avg_greedy_score = np.mean(greedy_rewards)
    
    print("\n" + "="*40)
    print("Baseline Agent Performance Comparison")
    print("="*40)
    print(f"Number of episodes per agent: {NUM_EPISODES}")
    print(f"Random Agent Average Score: {avg_random_score:.2f}")
    print(f"Greedy Agent Average Score: {avg_greedy_score:.2f}")
    print("="*40)