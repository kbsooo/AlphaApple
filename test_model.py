# 간단한 모델 테스트
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from train.train_maskable_ppo import make_env

# PPO 모델 로드
eval_env = DummyVecEnv([make_env(seed=999, rows=17, cols=10)])
model = PPO.load("ckpts/best_model.zip", env=eval_env)

# 한 에피소드 실행
obs = eval_env.reset()
total_reward = 0

for step in range(5):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step+1}: predicted action = {action}")
    obs, reward, done, info = eval_env.step(action)
    total_reward += reward[0]
    print(f"Reward: {reward[0]:.1f}, Done: {done[0]}")
    
    if done[0]:
        break

print(f"Total reward: {total_reward:.1f}")