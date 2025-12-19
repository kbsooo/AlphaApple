import torch
import torch.onnx
from src.models import FruitBoxDQN

def export_to_onnx(model_path, output_path, rows=10, cols=17):
    # Determine n_actions (same logic as env)
    n_actions = 0
    for r1 in range(rows):
        for r2 in range(r1, rows):
            for c1 in range(cols):
                for c2 in range(c1, cols):
                    n_actions += 1
    
    # Load model
    model = FruitBoxDQN(rows, cols, n_actions)
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'policy_net' in checkpoint:
        model.load_state_dict(checkpoint['policy_net'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create dummy input: (Batch, Channels, Rows, Cols)
    dummy_input = torch.zeros(1, 10, rows, cols)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output_path", type=str, default="model.onnx", help="Path to save .onnx model")
    args = parser.parse_args()
    
    export_to_onnx(args.model_path, args.output_path)
