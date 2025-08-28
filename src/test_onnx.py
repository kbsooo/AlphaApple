# test_onnx.py
"""
변환된 ONNX 모델을 테스트하고 원본 PyTorch 모델과 성능 비교
"""

import numpy as np
import onnxruntime as ort
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.fruitbox_env import FruitBoxEnv, FruitBoxConfig
from train.train_maskable_ppo import make_env, SmallGridCNN

class ONNXAgent:
    """ONNX 모델을 사용하는 에이전트"""
    
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        
    def predict(self, obs, deterministic=True):
        """관측을 받아 행동 예측"""
        # obs shape: (batch, 17, 10, 1) float32
        ort_inputs = {self.input_name: obs}
        outputs = self.session.run(None, ort_inputs)
        
        # SB3 정책의 출력이 여러 개일 수 있음. 첫 번째가 action logits
        action_logits = outputs[0]
        value = outputs[1] if len(outputs) > 1 else None
        
        # action_logits가 1D면 2D로 만들기
        if action_logits.ndim == 1:
            action_logits = action_logits[None, :]
        
        # 가장 높은 확률의 행동 선택 (deterministic)
        if deterministic:
            action = np.argmax(action_logits, axis=1)
        else:
            # 확률적 샘플링
            probs = np.exp(action_logits - np.max(action_logits, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            action = np.array([np.random.choice(len(p), p=p) for p in probs])
            
        return action, value

def evaluate_onnx_model(onnx_path="exports/fruitbox_ppo.onnx", num_episodes=20):
    """ONNX 모델 성능 평가"""
    
    agent = ONNXAgent(onnx_path)
    
    # 평가용 환경 (래퍼 적용해서 동일한 전처리)
    eval_env = DummyVecEnv([make_env(seed=999, rows=17, cols=10)])
    
    scores = []
    print(f"Testing ONNX model ({num_episodes} episodes)...")
    
    for ep in range(num_episodes):
        obs = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, value = agent.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
            
        scores.append(episode_reward)
        print(f"Episode {ep+1}: {episode_reward:.1f}")
    
    return np.array(scores)

def compare_pytorch_vs_onnx():
    """PyTorch vs ONNX 모델 성능 비교"""
    
    print("Loading PyTorch model...")
    eval_env = DummyVecEnv([make_env(seed=999, rows=17, cols=10)])
    pytorch_model = PPO.load(
        "ckpts/best_model.zip",
        env=eval_env,
        custom_objects={
            "policy_kwargs": dict(
                features_extractor_class=SmallGridCNN,
                features_extractor_kwargs=dict(features_dim=128)
            )
        }
    )
    
    print("Loading ONNX model...")
    onnx_agent = ONNXAgent("exports/fruitbox_ppo.onnx")
    
    # 같은 시드로 5 에피소드 비교
    n_test = 5
    pytorch_scores = []
    onnx_scores = []
    
    for ep in range(n_test):
        # PyTorch 평가
        obs = eval_env.reset()  # 고정 시드 환경
        pytorch_reward = 0
        done = False
        while not done:
            action, _ = pytorch_model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            pytorch_reward += reward[0]
        pytorch_scores.append(pytorch_reward)
        
        # ONNX 평가 (같은 환경)
        obs = eval_env.reset()
        onnx_reward = 0
        done = False
        while not done:
            action, _ = onnx_agent.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            onnx_reward += reward[0]
        onnx_scores.append(onnx_reward)
        
        print(f"Episode {ep+1}: PyTorch={pytorch_reward:.1f}, ONNX={onnx_reward:.1f}")
    
    print("\n" + "="*40)
    print("PYTORCH vs ONNX COMPARISON")
    print("="*40)
    print(f"PyTorch avg: {np.mean(pytorch_scores):.2f}")
    print(f"ONNX avg:    {np.mean(onnx_scores):.2f}")
    print(f"Difference:  {abs(np.mean(pytorch_scores) - np.mean(onnx_scores)):.2f}")

if __name__ == "__main__":
    # ONNX 모델 단독 평가
    onnx_scores = evaluate_onnx_model()
    print(f"\nONNX Model Average: {onnx_scores.mean():.2f} ± {onnx_scores.std():.2f}")
    
    # PyTorch vs ONNX 비교
    print("\n" + "="*50)
    compare_pytorch_vs_onnx()