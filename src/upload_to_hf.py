# upload_to_hf.py
"""
학습된 모델을 Hugging Face Hub에 업로드하는 스크립트
PyTorch 모델과 ONNX 모델을 모두 업로드
"""

import os
import json
import shutil
from pathlib import Path

try:
    from huggingface_hub import HfApi, Repository, upload_file, create_repo
    HF_AVAILABLE = True
except ImportError:
    print("Warning: huggingface_hub not installed. Run: uv add huggingface_hub")
    HF_AVAILABLE = False

def create_model_card(repo_name="fruitbox-ppo-agent"):
    """모델 카드 (README.md) 생성"""
    
    model_card = f"""---
library_name: stable-baselines3
tags:
- FruitBox
- reinforcement-learning
- ppo
- game-ai
- puzzle-solving
model-index:
- name: {repo_name}
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
import {{ InferenceSession }} from 'onnxruntime-web';

// Load ONNX model
const session = await InferenceSession.create('./fruitbox_ppo.onnx');

// Predict action
const {{ action_logits }} = await session.run({{
    board_input: new ort.Tensor('float32', board_data, [1, 17, 10, 1])
}});
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
@misc{{alphaapple2024,
  title={{AlphaApple: AI Agent for FruitBox Puzzle Game}},
  author={{Your Name}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/{repo_name}}}}}
}}
```
"""
    
    return model_card

def upload_to_huggingface(
    repo_name="fruitbox-ppo-agent",
    username=None,  # 사용자가 직접 입력
    private=False
):
    """모델을 Hugging Face Hub에 업로드"""
    
    if not HF_AVAILABLE:
        print("Error: huggingface_hub not installed")
        print("Install with: uv add huggingface_hub")
        return False
    
    if username is None:
        username = input("Enter your Hugging Face username: ")
        
    repo_id = f"{username}/{repo_name}"
    
    print(f"Creating repository: {repo_id}")
    
    # 레포지토리 생성 (이미 있으면 스킵)
    api = HfApi()
    try:
        create_repo(repo_id, exist_ok=True, private=private)
        print(f"Repository {repo_id} ready!")
    except Exception as e:
        print(f"Repository creation failed: {e}")
        return False
    
    # 업로드할 파일들 준비
    upload_files = []
    
    # 1. PyTorch 모델
    if os.path.exists("ckpts/best_model.zip"):
        upload_files.append(("ckpts/best_model.zip", "pytorch_model.zip"))
    
    # 2. ONNX 모델
    if os.path.exists("exports/fruitbox_ppo.onnx"):
        upload_files.append(("exports/fruitbox_ppo.onnx", "fruitbox_ppo.onnx"))
    
    # 3. 모델 정보
    if os.path.exists("exports/model_info.json"):
        upload_files.append(("exports/model_info.json", "model_info.json"))
    
    # 4. README (모델 카드) 생성
    readme_content = create_model_card(repo_name)
    with open("exports/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    upload_files.append(("exports/README.md", "README.md"))
    
    # 파일 업로드
    print("Uploading files...")
    for local_path, remote_name in upload_files:
        try:
            print(f"  Uploading {remote_name}...")
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"  ✓ {remote_name} uploaded")
        except Exception as e:
            print(f"  ✗ Failed to upload {remote_name}: {e}")
    
    print("\n" + "="*50)
    print("HUGGING FACE UPLOAD COMPLETE!")
    print("="*50)
    print(f"Model URL: https://huggingface.co/{repo_id}")
    print(f"ONNX Model: https://huggingface.co/{repo_id}/resolve/main/fruitbox_ppo.onnx")
    print("="*50)
    
    return True

def prepare_upload_files():
    """업로드 전 파일들 확인 및 준비"""
    
    required_files = [
        ("ckpts/best_model.zip", "PyTorch 모델"),
        ("exports/fruitbox_ppo.onnx", "ONNX 모델"),
        ("exports/model_info.json", "모델 메타데이터")
    ]
    
    print("Checking required files...")
    missing = []
    
    for file_path, description in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  ✓ {description}: {file_path} ({size:.2f} MB)")
        else:
            print(f"  ✗ Missing: {description} ({file_path})")
            missing.append(file_path)
    
    if missing:
        print(f"\nError: Missing files: {missing}")
        print("Run the following to generate missing files:")
        if "exports/fruitbox_ppo.onnx" in missing:
            print("  python src/convert_to_onnx.py")
        return False
    
    print("\nAll required files are ready for upload!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-name", default="fruitbox-ppo-agent", help="Repository name")
    parser.add_argument("--username", help="Hugging Face username")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--check-only", action="store_true", help="Only check files, don't upload")
    
    args = parser.parse_args()
    
    # 파일 확인
    if not prepare_upload_files():
        exit(1)
    
    if args.check_only:
        print("File check complete. Use --username to proceed with upload.")
        exit(0)
    
    # Hugging Face 업로드
    if not HF_AVAILABLE:
        print("Install huggingface_hub first: uv add huggingface_hub")
        exit(1)
        
    success = upload_to_huggingface(
        repo_name=args.repo_name,
        username=args.username, 
        private=args.private
    )
    
    if success:
        print("\n🎉 Upload successful!")
    else:
        print("\n❌ Upload failed!")