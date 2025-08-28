# full_comparison.py
"""
모든 에이전트 (Random, Greedy, PPO) 성능 비교
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.fruitbox_env import FruitBoxEnv, FruitBoxConfig
from train.train_maskable_ppo import make_env, SmallGridCNN
from src.baseline_eval import random_policy, greedy_policy, play_episode

def evaluate_all_agents(num_episodes=100):
    """모든 에이전트 성능 비교"""
    
    # 베이스라인용 환경 (학습과 동일한 크기)
    cfg = FruitBoxConfig(rows=17, cols=10, render_mode=None)
    base_env = FruitBoxEnv(cfg)
    
    # PPO용 환경 (학습과 동일한 크기)
    eval_env = DummyVecEnv([make_env(seed=999, rows=17, cols=10)])
    
    print("Loading PPO model...")
    try:
        ppo_model = PPO.load(
            "ckpts/best_model.zip", 
            env=eval_env,
            custom_objects={
                "policy_kwargs": dict(
                    features_extractor_class=SmallGridCNN,
                    features_extractor_kwargs=dict(features_dim=128)
                )
            }
        )
        print("PPO model loaded!")
    except Exception as e:
        print(f"Failed to load PPO model: {e}")
        return
    
    # 베이스라인 평가
    print(f"\nEvaluating agents ({num_episodes} episodes each)...")
    
    def run_baseline(policy, name):
        scores = [play_episode(base_env, policy) for _ in range(num_episodes)]
        return np.array(scores)
    
    random_scores = run_baseline(random_policy, "Random")
    greedy_scores = run_baseline(greedy_policy, "Greedy")
    
    # PPO 평가
    ppo_scores = []
    for _ in range(num_episodes):
        obs = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
            
        ppo_scores.append(episode_reward)
    
    ppo_scores = np.array(ppo_scores)
    
    # 결과 출력
    print("\n" + "="*60)
    print("AGENT PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Agent':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-"*60)
    print(f"{'Random':<12} {random_scores.mean():<8.2f} {random_scores.std():<8.2f} {random_scores.min():<8.1f} {random_scores.max():<8.1f}")
    print(f"{'Greedy':<12} {greedy_scores.mean():<8.2f} {greedy_scores.std():<8.2f} {greedy_scores.min():<8.1f} {greedy_scores.max():<8.1f}")
    print(f"{'PPO (AI)':<12} {ppo_scores.mean():<8.2f} {ppo_scores.std():<8.2f} {ppo_scores.min():<8.1f} {ppo_scores.max():<8.1f}")
    print("="*60)
    
    # 개선율 계산
    ppo_vs_random = (ppo_scores.mean() / random_scores.mean() - 1) * 100
    ppo_vs_greedy = (ppo_scores.mean() / greedy_scores.mean() - 1) * 100
    
    print(f"\nPPO vs Random: {ppo_vs_random:+.1f}% improvement")
    print(f"PPO vs Greedy: {ppo_vs_greedy:+.1f}% improvement")
    
    return {
        'random': random_scores,
        'greedy': greedy_scores, 
        'ppo': ppo_scores
    }

if __name__ == "__main__":
    results = evaluate_all_agents(100)