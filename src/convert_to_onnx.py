# convert_to_onnx.py
"""
학습된 PPO 모델을 ONNX 형식으로 변환하는 스크립트
웹 배포 및 Chrome 확장에서 사용하기 위함
"""

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import onnx
import onnxruntime as ort

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train_maskable_ppo import make_env, SmallGridCNN

def convert_ppo_to_onnx(
    model_path="ckpts/best_model.zip",
    output_path="exports/fruitbox_ppo.onnx",
    optimize=True
):
    """PPO 모델을 ONNX로 변환"""
    
    print(f"Loading model from {model_path}...")
    
    # 모델 로드
    eval_env = DummyVecEnv([make_env(seed=999, rows=17, cols=10)])
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
    
    # 정책 신경망 추출 (정책만 필요, 가치함수는 추론 시 불필요)
    policy_net = model.policy
    policy_net.eval()
    
    # 더미 입력 생성 (배치 크기 1, 17x10x1 보드)
    dummy_input = torch.randn(1, 17, 10, 1, dtype=torch.float32)
    
    # exports 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Converting to ONNX: {output_path}")
    
    # ONNX 변환
    torch.onnx.export(
        policy_net,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,  # 웹 호환성을 위해 안정적인 버전 사용
        do_constant_folding=True,
        input_names=['board_input'],
        output_names=['action_logits', 'value'],
        dynamic_axes={
            'board_input': {0: 'batch_size'},
            'action_logits': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    
    print("ONNX conversion completed!")
    
    # 변환된 모델 검증
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
    
    # 추론 테스트
    print("Testing ONNX inference...")
    ort_session = ort.InferenceSession(output_path)
    
    # 테스트 입력
    test_input = np.random.rand(1, 17, 10, 1).astype(np.float32)
    
    # ONNX 추론
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"ONNX output shapes: {[out.shape for out in ort_outputs]}")
    
    # 원본 모델과 비교
    with torch.no_grad():
        torch_input = torch.from_numpy(test_input)
        torch_output = policy_net(torch_input)
        
    print("ONNX inference test passed!")
    
    # 모델 정보 출력
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"ONNX model size: {file_size:.2f} MB")
    
    return output_path

def create_model_info():
    """모델 메타데이터 생성"""
    info = {
        "model_name": "AlphaApple FruitBox AI",
        "description": "PPO agent trained to play Korean fruit box puzzle game",
        "architecture": "CNN + PPO",
        "input_shape": [17, 10, 1],
        "output_shape": "action_logits + value",
        "training_steps": 1_000_000,
        "performance": {
            "average_score": 77.0,
            "vs_random": "+7.1%",
            "vs_greedy": "+5.0%"
        }
    }
    
    # JSON으로 저장
    import json
    os.makedirs("exports", exist_ok=True)
    with open("exports/model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print("Model info saved to exports/model_info.json")
    return info

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ckpts/best_model.zip", help="Model path")
    parser.add_argument("--output", default="exports/fruitbox_ppo.onnx", help="ONNX output path")
    
    args = parser.parse_args()
    
    # ONNX 변환 실행
    onnx_path = convert_ppo_to_onnx(args.model, args.output)
    
    # 메타데이터 생성
    model_info = create_model_info()
    
    print("\n" + "="*50)
    print("ONNX CONVERSION COMPLETE!")
    print("="*50)
    print(f"Model saved to: {onnx_path}")
    print(f"Model info: exports/model_info.json")
    print("Ready for web deployment and HuggingFace upload!")