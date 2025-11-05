"""
Generator-Solver Paired Learning

아이디어:
- Generator: "풀 수 있는" 보드를 생성 (목표: Solver가 높은 clearance 달성)
- Solver: 보드를 최대한 잘 풀기 (목표: clearance 최대화)

이 둘을 co-evolution 방식으로 학습:
1. Generator가 보드 생성
2. Solver가 플레이
3. Generator는 Solver의 성능을 보상으로 받음
4. Solver는 일반적인 RL 보상
"""

import torch
import torch.nn as nn
import numpy as np


class BoardGeneratorNetwork(nn.Module):
    """
    신경망 기반 보드 생성기

    Input: noise vector (latent z) + target_clearance
    Output: 10×17 보드 (숫자 0-9)
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Decoder: latent → 10×17 보드
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),  # +1 for target_clearance
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 10 * 17 * 10),  # 10×17 cells, 10 classes (0-9)
        )

    def forward(self, z, target_clearance):
        """
        Args:
            z: (batch, latent_dim) noise vector
            target_clearance: (batch, 1) 목표 clearance (0-1)

        Returns:
            board: (batch, 10, 17) 보드
        """
        x = torch.cat([z, target_clearance], dim=-1)
        logits = self.decoder(x).view(-1, 10, 17, 10)  # (B, 10, 17, 10)

        # Softmax로 0-9 확률 분포
        probs = torch.softmax(logits, dim=-1)

        # Gumbel-Softmax로 미분 가능하게 샘플링
        board = torch.argmax(probs, dim=-1)  # (B, 10, 17)

        return board, probs


class SolverNetwork(nn.Module):
    """
    보드를 푸는 RL 에이전트 (Policy)
    """
    def __init__(self):
        super().__init__()
        # ... (이전에 제안한 Autoregressive 또는 Maskable PPO)
        pass


class PairedLearning:
    """
    Generator와 Solver를 함께 학습
    """
    def __init__(self):
        self.generator = BoardGeneratorNetwork()
        self.solver = SolverNetwork()

        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.solver_optimizer = torch.optim.Adam(self.solver.parameters(), lr=3e-4)

    def train_iteration(self, batch_size=32, target_clearance=0.7):
        """
        1회 학습 반복
        """
        # === Phase 1: Generator로 보드 생성 ===
        z = torch.randn(batch_size, self.generator.latent_dim)
        target = torch.ones(batch_size, 1) * target_clearance

        boards, probs = self.generator(z, target)

        # === Phase 2: Solver로 플레이 ===
        clearances = []
        for board in boards:
            # Solver가 이 보드를 플레이
            board_np = board.detach().cpu().numpy()
            clearance, actions, rewards = self.solver.play_episode(board_np)
            clearances.append(clearance)

            # Solver 업데이트 (일반적인 RL)
            self.solver.update(board_np, actions, rewards)

        clearances = torch.tensor(clearances)

        # === Phase 3: Generator 업데이트 ===
        # 목표 1: Solver가 높은 clearance 달성
        clearance_reward = clearances.mean()

        # 목표 2: target_clearance에 가깝게 (너무 쉽거나 어렵지 않게)
        clearance_loss = ((clearances - target.squeeze()) ** 2).mean()

        # 목표 3: 다양성 (entropy 최대화)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        # 총 Generator 손실
        gen_loss = -clearance_reward + clearance_loss * 0.5 - entropy * 0.01

        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        return {
            'gen_loss': gen_loss.item(),
            'avg_clearance': clearances.mean().item(),
            'clearance_std': clearances.std().item(),
        }


# ============================================================
# 실제 사용 예제
# ============================================================

if __name__ == "__main__":
    # Paired Learning 초기화
    paired = PairedLearning()

    # 학습 루프
    for iteration in range(10000):
        stats = paired.train_iteration(
            batch_size=32,
            target_clearance=0.7  # 70% 제거 가능한 보드 생성 목표
        )

        if iteration % 100 == 0:
            print(f"Iter {iteration}: "
                  f"Clearance {stats['avg_clearance']:.1%} ± {stats['clearance_std']:.1%}, "
                  f"Loss {stats['gen_loss']:.3f}")

    # 학습 후 보드 생성
    z = torch.randn(1, 128)
    target = torch.tensor([[0.7]])
    board, _ = paired.generator(z, target)

    print("Generated board:")
    print(board[0].numpy())
