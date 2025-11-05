"""
전문가 데이터 수집

"작은 것 우선" 휴리스틱으로 에피소드를 플레이하고
Behavior Cloning 학습을 위한 데이터를 수집합니다.
"""

import numpy as np
import pickle
from tqdm import tqdm
from typing import List, Dict, Tuple
import sys
sys.path.insert(0, '/home/user/AlphaApple')

from envs.fruitbox_env import FruitBoxEnv, FruitBoxConfig
from envs.autoregressive_wrapper import make_autoregressive_env


def small_first_strategy(env: FruitBoxEnv, obs: np.ndarray) -> Tuple[int, int, int, int]:
    """
    작은 것 우선 전략: 가장 작은 직사각형 선택

    Returns:
        (r1, c1, r2, c2) 좌표
    """
    legal = env.legal_actions()

    if len(legal) == 0:
        return None

    # 가장 작은 직사각형 찾기
    sizes = []
    coords = []

    for action_idx in legal:
        r1, c1, r2, c2 = env.rects[action_idx]
        size = (r2 - r1 + 1) * (c2 - c1 + 1)
        sizes.append(size)
        coords.append((r1, c1, r2, c2))

    # 최소 크기 선택
    min_idx = np.argmin(sizes)
    return coords[min_idx]


def collect_expert_episodes(
    n_episodes: int = 1000,
    seed_start: int = 0,
    verbose: bool = True
) -> List[Dict]:
    """
    전문가 에피소드 수집

    Args:
        n_episodes: 수집할 에피소드 수
        seed_start: 시작 시드
        verbose: 진행상황 출력

    Returns:
        episodes: 에피소드 리스트
            각 에피소드는 {'observations': [...], 'actions': [...], 'reward': float}
    """
    episodes = []
    total_rewards = []

    iterator = range(n_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Collecting expert data")

    for i in iterator:
        seed = seed_start + i

        # 환경 초기화
        wrapped_env = make_autoregressive_env(rows=10, cols=17)
        env = wrapped_env.env
        obs, info = wrapped_env.reset(seed=seed)

        observations = []
        actions = []  # (r1, c1, r2, c2) 튜플들
        rewards = []

        episode_reward = 0
        steps = 0

        # 에피소드 플레이
        while True:
            # 현재 상태 저장
            observations.append(obs.copy())

            # 전문가 행동 선택
            coords = small_first_strategy(env, obs)

            if coords is None:
                # 더 이상 합법 행동 없음
                break

            r1, c1, r2, c2 = coords
            actions.append((r1, c1, r2, c2))

            # Step 실행
            obs, reward, terminated, truncated, info = wrapped_env.step_with_coords(r1, c1, r2, c2)

            rewards.append(reward)
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

            # 안전장치
            if steps >= 500:
                break

        # 에피소드 저장
        episodes.append({
            'observations': np.array(observations),  # (T, 10, 17)
            'actions': np.array(actions),            # (T, 4)
            'rewards': np.array(rewards),            # (T,)
            'total_reward': episode_reward,
            'steps': steps,
            'seed': seed,
        })

        total_rewards.append(episode_reward)

    # 통계
    if verbose:
        print(f"\n=== 수집 완료 ===")
        print(f"에피소드 수: {n_episodes}")
        print(f"평균 보상: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
        print(f"보상 범위: [{min(total_rewards):.0f}, {max(total_rewards):.0f}]")
        print(f"평균 스텝: {np.mean([ep['steps'] for ep in episodes]):.1f}")
        print(f"총 transition 수: {sum(ep['steps'] for ep in episodes)}")

    return episodes


def save_expert_data(episodes: List[Dict], filepath: str):
    """전문가 데이터 저장"""
    with open(filepath, 'wb') as f:
        pickle.dump(episodes, f)
    print(f"✅ 저장 완료: {filepath}")


def load_expert_data(filepath: str) -> List[Dict]:
    """전문가 데이터 로드"""
    with open(filepath, 'rb') as f:
        episodes = pickle.load(f)
    return episodes


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="전문가 데이터 수집")
    parser.add_argument('--n_episodes', type=int, default=1000,
                        help='수집할 에피소드 수')
    parser.add_argument('--seed_start', type=int, default=0,
                        help='시작 시드')
    parser.add_argument('--output', type=str, default='data/expert_data.pkl',
                        help='출력 파일 경로')
    parser.add_argument('--test', action='store_true',
                        help='테스트 모드 (10개만 수집)')

    args = parser.parse_args()

    if args.test:
        print("=== 테스트 모드 ===")
        n_episodes = 10
    else:
        n_episodes = args.n_episodes

    print(f"전문가 데이터 수집 시작...")
    print(f"  에피소드 수: {n_episodes}")
    print(f"  시작 시드: {args.seed_start}")
    print(f"  출력 파일: {args.output}")
    print()

    # 데이터 수집
    episodes = collect_expert_episodes(
        n_episodes=n_episodes,
        seed_start=args.seed_start,
        verbose=True
    )

    # 저장
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_expert_data(episodes, args.output)

    # 검증
    print(f"\n=== 데이터 검증 ===")
    loaded = load_expert_data(args.output)
    print(f"로드된 에피소드 수: {len(loaded)}")
    print(f"첫 번째 에피소드:")
    print(f"  Observations shape: {loaded[0]['observations'].shape}")
    print(f"  Actions shape: {loaded[0]['actions'].shape}")
    print(f"  Total reward: {loaded[0]['total_reward']}")
    print(f"  Steps: {loaded[0]['steps']}")
