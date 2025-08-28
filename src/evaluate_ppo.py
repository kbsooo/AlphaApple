# evaluate_ppo.py
"""
학습된 PPO 모델을 로드하고 평가하는 스크립트
"""

import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.fruitbox_env import FruitBoxEnv, FruitBoxConfig
from train.train_maskable_ppo import make_env, SmallGridCNN

def load_best_model(model_path="ckpts/best_model.zip"):
    """베스트 모델 로드"""
    print(f"Loading model from {model_path}...")
    
    # 평가용 환경 생성 (학습과 동일한 래퍼 구조)
    eval_env = DummyVecEnv([make_env(seed=999)])
    
    # 모델 로드 (custom feature extractor 포함)
    model = PPO.load(
        model_path, 
        env=eval_env,
        custom_objects={
            "policy_kwargs": dict(
                features_extractor_class=SmallGridCNN,
                features_extractor_kwargs=dict(features_dim=128)
            )
        }
    )
    
    print("Model loaded successfully!")
    return model, eval_env

def evaluate_ppo_agent(model, env, num_episodes=100):
    """PPO 에이전트 평가"""
    total_rewards = []
    print(f"Evaluating PPO Agent ({num_episodes} episodes)...")
    
    for _ in tqdm(range(num_episodes)):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]  # VecEnv returns arrays
            
        total_rewards.append(episode_reward)
    
    return total_rewards

def play_interactive_game(model_path="ckpts/best_model.zip"):
    """인터랙티브하게 모델이 게임하는 모습 보기"""
    model, _ = load_best_model(model_path)
    
    # VecEnv로 래핑된 환경 사용 (학습과 동일한 환경)
    display_env = DummyVecEnv([make_env(seed=42, rows=17, cols=10)])
    
    obs = display_env.reset()
    print("\n=== PPO Agent Playing ===")
    
    total_reward = 0
    step = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = display_env.step(action)
        
        total_reward += reward[0]
        step += 1
        
        print(f"Step {step}: Action={action[0]}, Reward={reward[0]:.1f}")
        
        if done[0]:
            print("Game finished!")
            break
            
        if step >= 10:  # 10스텝만 보여주기
            print("(Demo complete)")
            break
    
    print(f"\nTotal Score: {total_reward:.1f}")
    return total_reward

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ckpts/best_model.zip", help="Model path")
    parser.add_argument("--interactive", action="store_true", help="Play interactive game")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    if args.interactive:
        play_interactive_game(args.model)
    else:
        model, eval_env = load_best_model(args.model)
        ppo_scores = evaluate_ppo_agent(model, eval_env, args.episodes)
        
        print("\n" + "="*50)
        print("PPO Agent Evaluation Results")
        print("="*50)
        print(f"Episodes: {len(ppo_scores)}")
        print(f"Average Score: {np.mean(ppo_scores):.2f} ± {np.std(ppo_scores):.2f}")
        print(f"Best Score: {np.max(ppo_scores):.2f}")
        print(f"Worst Score: {np.min(ppo_scores):.2f}")
        print("="*50)