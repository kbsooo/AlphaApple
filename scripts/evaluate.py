"""
학습된 정책 평가

Autoregressive Policy를 환경에서 실행하고 성능을 측정합니다.
"""

import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/user/AlphaApple')

from models.autoregressive_policy import AutoregressiveRectPolicy
from envs.autoregressive_wrapper import make_autoregressive_env


def evaluate_policy(
    policy: AutoregressiveRectPolicy,
    n_episodes: int = 100,
    seed_start: int = 10000,
    deterministic: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
):
    """
    정책 평가

    Args:
        policy: 평가할 정책
        n_episodes: 평가 에피소드 수
        seed_start: 시작 시드
        deterministic: True면 argmax, False면 샘플링
        device: 디바이스
        verbose: 진행상황 출력

    Returns:
        results: 결과 딕셔너리
    """
    policy = policy.to(device)
    policy.eval()

    episode_rewards = []
    episode_lengths = []
    illegal_action_counts = []

    iterator = range(n_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Evaluating")

    with torch.no_grad():
        for i in iterator:
            seed = seed_start + i

            # 환경 초기화
            env = make_autoregressive_env(rows=10, cols=17)
            obs, info = env.reset(seed=seed)

            episode_reward = 0
            steps = 0
            illegal_count = 0

            while True:
                # 관찰을 텐서로 변환
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 10, 17)
                obs_tensor = obs_tensor.to(device)

                # 정책으로 행동 선택
                action_tuple, log_prob, value, info_policy = policy(obs_tensor, deterministic=deterministic)

                r1 = int(action_tuple[0][0].item())
                c1 = int(action_tuple[1][0].item())
                r2 = int(action_tuple[2][0].item())
                c2 = int(action_tuple[3][0].item())

                # Step 실행
                obs, reward, terminated, truncated, info_env = env.step_with_coords(r1, c1, r2, c2)

                # 불법 행동 카운트
                if info_env.get('illegal_action', False):
                    illegal_count += 1

                episode_reward += reward
                steps += 1

                if terminated or truncated:
                    break

                # 안전장치
                if steps >= 500:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            illegal_action_counts.append(illegal_count)

    # 통계
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_illegal': np.mean(illegal_action_counts),
        'episode_rewards': episode_rewards,
    }

    if verbose:
        print(f"\n=== 평가 결과 ===")
        print(f"에피소드 수: {n_episodes}")
        print(f"평균 보상: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
        print(f"보상 범위: [{results['min_reward']:.0f}, {results['max_reward']:.0f}]")
        print(f"평균 스텝: {results['mean_length']:.1f}")
        print(f"평균 불법 행동: {results['mean_illegal']:.2f}")
        print(f"\n클리어 비율: {results['mean_reward'] / 170 * 100:.1f}%")

    return results


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="정책 평가")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='모델 체크포인트 경로')
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='평가 에피소드 수')
    parser.add_argument('--seed_start', type=int, default=10000,
                        help='시작 시드')
    parser.add_argument('--stochastic', action='store_true',
                        help='Stochastic 모드 (샘플링)')

    args = parser.parse_args()

    print(f"=== 정책 평가 ===")
    print(f"체크포인트: {args.checkpoint}")
    print(f"에피소드: {args.n_episodes}")
    print(f"모드: {'Stochastic' if args.stochastic else 'Deterministic'}")
    print()

    # 모델 로드
    print("모델 로딩...")
    policy = AutoregressiveRectPolicy(rows=10, cols=17, latent_dim=256)

    try:
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        policy.load_state_dict(state_dict)
        print(f"✅ 체크포인트 로드 완료")
    except Exception as e:
        print(f"❌ 체크포인트 로드 실패: {e}")
        print("랜덤 초기화 상태로 평가합니다.")

    print()

    # 평가
    results = evaluate_policy(
        policy=policy,
        n_episodes=args.n_episodes,
        seed_start=args.seed_start,
        deterministic=not args.stochastic,
        verbose=True
    )

    # 베이스라인과 비교
    print(f"\n=== 베이스라인 비교 ===")
    print(f"Greedy (작은 것): 105.3개 (62.0%)")
    print(f"사람 (당신):      ~115개 (67.6%)")
    print(f"현재 정책:        {results['mean_reward']:.1f}개 ({results['mean_reward']/170*100:.1f}%)")
