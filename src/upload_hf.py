from huggingface_hub import HfApi, create_repo
import os

def upload_to_hf(repo_id, model_path, onnx_path=None, README_path=None):
    api = HfApi()
    
    # Create repo if not exists
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"Repo {repo_id} ready.")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Upload main model
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.pth",
        repo_id=repo_id,
    )
    print(f"Uploaded {model_path} to {repo_id}")

    # Upload ONNX model if provided
    if onnx_path and os.path.exists(onnx_path):
        api.upload_file(
            path_or_fileobj=onnx_path,
            path_in_repo="model.onnx",
            repo_id=repo_id,
        )
        print(f"Uploaded {onnx_path} to {repo_id}")

    # Upload README
    if README_path and os.path.exists(README_path):
        api.upload_file(
            path_or_fileobj=README_path,
            path_in_repo="README.md",
            repo_id=repo_id,
        )
        print(f"Uploaded {README_path} to {repo_id}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID (e.g., username/repo)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to PyTorch .pth file")
    parser.add_argument("--onnx_path", type=str, help="Path to converted .onnx file")
    parser.add_argument("--readme_path", type=str, help="Path to README.md")
    args = parser.parse_args()
    
    # Needs huggingface-cli login before running
    upload_to_hf(args.repo_id, args.model_path, args.onnx_path, args.readme_path)
