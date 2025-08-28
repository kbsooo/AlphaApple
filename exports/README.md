---
library_name: stable-baselines3
tags:
- FruitBox
- reinforcement-learning
- ppo
- game-ai
- puzzle-solving
model-index:
- name: AlphaApple
  results:
  - task:
      type: reinforcement-learning
      name: Reinforcement Learning
    dataset:
      name: FruitBox Game
      type: fruitbox
    metrics:
    - type: mean_reward
      value: 77.0
      name: Mean Episode Score
    - type: improvement_vs_random
      value: 7.1%
      name: Improvement vs Random
    - type: improvement_vs_greedy  
      value: 5.0%
      name: Improvement vs Greedy
---

# AlphaApple: FruitBox Game AI Agent

## Model Description

이 모델은 한국의 사과게임(FruitBox) 퍼즐을 해결하는 AI 에이전트입니다. 
10×17 격자에서 합이 10인 직사각형을 찾아 제거하는 게임을 PPO(Proximal Policy Optimization) 알고리즘으로 학습했습니다.

## Game Rules

- 10×17 격자, 각 셀은 1-9 숫자
- 직사각형 영역을 선택해서 숫자 합이 정확히 10이면 해당 영역 제거
- 제거된 셀 개수만큼 점수 획득
- 더 이상 제거할 수 있는 영역이 없으면 게임 종료

## Performance

| Agent   | Average Score | Improvement |
|---------|--------------|-------------|
| Random  | 71.9         | -           |
| Greedy  | 73.3         | +1.9%       |
| **PPO** | **77.0**     | **+7.1%**   |

## Usage

### Python (PyTorch)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Load model
model = PPO.load("pytorch_model.zip")

# Use for inference
obs = env.reset()
action, _ = model.predict(obs)
```

### Web/JavaScript (ONNX)

```javascript
import { InferenceSession } from 'onnxruntime-web';

// Load ONNX model
const session = await InferenceSession.create('./fruitbox_ppo.onnx');

// Predict action
const { action_logits } = await session.run({
    board_input: new ort.Tensor('float32', board_data, [1, 17, 10, 1])
});
const action = action_logits.data.indexOf(Math.max(...action_logits.data));
```

## Files

- `pytorch_model.zip`: Original SB3 PPO model 
- `fruitbox_ppo.onnx`: ONNX version for web deployment (2.95MB)
- `model_info.json`: Model metadata and performance metrics

## Training Details

- Algorithm: PPO with action masking
- Network: Custom CNN (SmallGridCNN)
- Training steps: 1,000,000
- Environment: Custom Gymnasium environment
- Action space: 8,415 possible rectangles (masked)

## Repository

Source code: https://github.com/your-username/alphaapple

## Citation

```bibtex
@misc{alphaapple2024,
  title={AlphaApple: AI Agent for FruitBox Puzzle Game},
  author={Your Name},
  year={2024},
  howpublished={\url{https://huggingface.co/AlphaApple}}
}
```
