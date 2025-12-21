---
library_name: pytorch
pipeline_tag: reinforcement-learning
tags:
- reinforcement-learning
- dqn
- onnx
- fruitbox
- gamesaien
- chrome-extension
- gymnasium
---

# AlphaApple - FruitBox DQN

This model plays the FruitBox (Fruit Box) puzzle game hosted on Gamesaien. It predicts Q-values over all axis-aligned rectangles on a 10x17 board. A valid action is a rectangle whose cell sum is exactly 10, so you must apply an action mask to filter invalid rectangles before selecting the best move.

## Quick facts
- Board: 10x17, values 0-9 (0 means empty)
- Action space: 8415 axis-aligned rectangles
- Input: one-hot board with shape `[1, 10, 10, 17]`
- Output: Q-values for all rectangles
- Masking: required to remove invalid rectangles

## Files in this repo
- `model.pth`: PyTorch checkpoint dict with `policy_net`, `target_net`, `optimizer`
- `model.onnx`: Exported ONNX model for browser/runtime inference

## How to use (PyTorch)
```python
# Model definition is in https://github.com/kbsooo/AlphaApple (src/models.py)
import torch
from src.models import FruitBoxDQN

rows, cols = 10, 17
n_actions = 55 * 153  # (rows*(rows+1)/2) * (cols*(cols+1)/2) = 8415
model = FruitBoxDQN(rows, cols, n_actions)

ckpt = torch.load("model.pth", map_location="cpu")
state = ckpt["policy_net"] if "policy_net" in ckpt else ckpt
model.load_state_dict(state)
model.eval()
```

## How to use (ONNX / browser)
```js
const session = await ort.InferenceSession.create("model.onnx");
// input: Float32Array with shape [1, 10, 10, 17]
const output = await session.run({ input });
// output.output.data: Q-values for 8415 rectangles
```

## Action masking (required)
You must mask invalid rectangles before selecting an action. A rectangle is valid if the sum of its cells equals 10. Without the mask, the model can pick illegal moves.

## Training details
- Environment: FruitBoxEnv (implemented in `envs/fruitbox_env.py`, class `FruitBoxEnvImproved`)
- Board generator: BackwardBoardGenerator (solvable boards)
- Curriculum: target coverage ramps from 0.3 to 0.95 in steps of 0.1
- Optimizer: Adam, gamma=0.99, lr=1e-4
- Episodes: 10k (Colab integrated script)

## Limitations
- Trained on generated boards; performance may vary on edge cases.
- Requires an accurate action mask and correct board extraction.

## Links
- Game: https://en.gamesaien.com/game/fruit_box/
- Project: https://github.com/kbsooo/AlphaApple
