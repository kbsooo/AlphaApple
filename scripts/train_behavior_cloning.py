"""
Behavior Cloning (모방 학습)

전문가 데이터로 Autoregressive Policy를 사전학습합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/user/AlphaApple')

from models.autoregressive_policy import AutoregressiveRectPolicy
from scripts.collect_expert_data import load_expert_data


class ExpertDataset(Dataset):
    """전문가 데이터셋"""

    def __init__(self, episodes):
        """
        Args:
            episodes: collect_expert_data.py로 수집한 에피소드 리스트
        """
        self.observations = []
        self.actions = []

        # 모든 에피소드의 transition 수집
        for ep in episodes:
            obs = ep['observations']  # (T, 10, 17)
            acts = ep['actions']      # (T, 4)

            for t in range(len(obs)):
                self.observations.append(obs[t])
                self.actions.append(acts[t])

        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)

        print(f"데이터셋 생성 완료:")
        print(f"  Transitions: {len(self)}")
        print(f"  Observations shape: {self.observations.shape}")
        print(f"  Actions shape: {self.actions.shape}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.observations[idx]).float()  # (10, 17)
        act = torch.from_numpy(self.actions[idx]).long()        # (4,) -> (r1, c1, r2, c2)

        # (10, 17) → (1, 10, 17)
        obs = obs.unsqueeze(0)

        return obs, act


def train_behavior_cloning(
    policy: AutoregressiveRectPolicy,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 50,
    lr: float = 3e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_path: str = 'checkpoints/bc_policy.pt'
):
    """
    Behavior Cloning 학습

    Args:
        policy: Autoregressive Policy 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        n_epochs: 학습 에폭 수
        lr: 학습률
        device: 디바이스
        save_path: 모델 저장 경로
    """
    policy = policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"\n=== Behavior Cloning 학습 시작 ===")
    print(f"Device: {device}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()

    for epoch in range(n_epochs):
        # ===== 학습 =====
        policy.train()
        train_loss_epoch = 0
        train_acc_epoch = [0, 0, 0, 0]  # r1, c1, r2, c2 각각

        for batch_obs, batch_act in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False):
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            # 전문가 행동
            r1_expert = batch_act[:, 0]
            c1_expert = batch_act[:, 1]
            r2_expert = batch_act[:, 2]
            c2_expert = batch_act[:, 3]

            # Forward pass (주어진 행동으로 log_prob 계산)
            action_tuple = (r1_expert, c1_expert, r2_expert, c2_expert)
            _, log_prob, _, info = policy(batch_obs, action=action_tuple)

            # 손실: negative log likelihood
            loss = -log_prob.mean()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

            # 정확도 계산 (각 좌표별)
            r1_pred = info['r1']
            c1_pred = info['c1']
            r2_pred = info['r2']
            c2_pred = info['c2']

            train_acc_epoch[0] += (r1_pred == r1_expert).float().mean().item()
            train_acc_epoch[1] += (c1_pred == c1_expert).float().mean().item()
            train_acc_epoch[2] += (r2_pred == r2_expert).float().mean().item()
            train_acc_epoch[3] += (c2_pred == c2_expert).float().mean().item()

        train_loss_epoch /= len(train_loader)
        train_acc_epoch = [acc / len(train_loader) for acc in train_acc_epoch]
        train_losses.append(train_loss_epoch)

        # ===== 검증 =====
        policy.eval()
        val_loss_epoch = 0
        val_acc_epoch = [0, 0, 0, 0]

        with torch.no_grad():
            for batch_obs, batch_act in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False):
                batch_obs = batch_obs.to(device)
                batch_act = batch_act.to(device)

                r1_expert = batch_act[:, 0]
                c1_expert = batch_act[:, 1]
                r2_expert = batch_act[:, 2]
                c2_expert = batch_act[:, 3]

                action_tuple = (r1_expert, c1_expert, r2_expert, c2_expert)
                _, log_prob, _, info = policy(batch_obs, action=action_tuple)

                loss = -log_prob.mean()
                val_loss_epoch += loss.item()

                r1_pred = info['r1']
                c1_pred = info['c1']
                r2_pred = info['r2']
                c2_pred = info['c2']

                val_acc_epoch[0] += (r1_pred == r1_expert).float().mean().item()
                val_acc_epoch[1] += (c1_pred == c1_expert).float().mean().item()
                val_acc_epoch[2] += (r2_pred == r2_expert).float().mean().item()
                val_acc_epoch[3] += (c2_pred == c2_expert).float().mean().item()

        val_loss_epoch /= len(val_loader)
        val_acc_epoch = [acc / len(val_loader) for acc in val_acc_epoch]
        val_losses.append(val_loss_epoch)

        # 출력
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {train_loss_epoch:.4f}, Acc: r1={train_acc_epoch[0]:.3f} c1={train_acc_epoch[1]:.3f} r2={train_acc_epoch[2]:.3f} c2={train_acc_epoch[3]:.3f}")
        print(f"  Val Loss:   {val_loss_epoch:.4f}, Acc: r1={val_acc_epoch[0]:.3f} c1={val_acc_epoch[1]:.3f} r2={val_acc_epoch[2]:.3f} c2={val_acc_epoch[3]:.3f}")

        # Best 모델 저장
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(policy.state_dict(), save_path)
            print(f"  ✅ Best model saved: {save_path}")

        print()

    print(f"\n=== 학습 완료 ===")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"모델 저장 경로: {save_path}")

    return train_losses, val_losses


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Behavior Cloning 학습")
    parser.add_argument('--data', type=str, default='data/expert_data_1000.pkl',
                        help='전문가 데이터 경로')
    parser.add_argument('--epochs', type=int, default=50,
                        help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='학습률')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='검증 데이터 비율')
    parser.add_argument('--save_path', type=str, default='checkpoints/bc_policy.pt',
                        help='모델 저장 경로')

    args = parser.parse_args()

    print(f"=== Behavior Cloning ===")
    print(f"데이터: {args.data}")
    print(f"에폭: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"학습률: {args.lr}")
    print()

    # 데이터 로드
    print("데이터 로딩...")
    episodes = load_expert_data(args.data)
    print(f"에피소드 수: {len(episodes)}")
    print(f"평균 보상: {np.mean([ep['total_reward'] for ep in episodes]):.1f}")
    print()

    # 데이터셋 생성
    dataset = ExpertDataset(episodes)

    # Train/Val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print()

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 모델 생성
    policy = AutoregressiveRectPolicy(rows=10, cols=17, latent_dim=256)
    print(f"모델 파라미터 수: {sum(p.numel() for p in policy.parameters()):,}")
    print()

    # 학습
    train_losses, val_losses = train_behavior_cloning(
        policy=policy,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path
    )
