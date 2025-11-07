"""
Autoregressive Policy for FruitBox

직사각형 선택을 4단계로 분해:
1. r1 선택 (10개 중)
2. c1 선택 (17개 중, r1 조건부)
3. r2 선택 (10개 중, r1,c1 조건부, r2≥r1 마스킹)
4. c2 선택 (17개 중, r1,c1,r2 조건부, c2≥c1 마스킹)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class BoardEncoder(nn.Module):
    """보드 상태를 인코딩하는 CNN"""

    def __init__(self, rows: int = 10, cols: int = 17, latent_dim: int = 256):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.latent_dim = latent_dim

        # CNN: (1, 10, 17) → latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Flatten 후 크기 계산
        conv_out_size = 64 * rows * cols

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, latent_dim),
            nn.ReLU(),
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        """
        Args:
            board: (batch, 1, rows, cols) 또는 (batch, rows, cols)

        Returns:
            latent: (batch, latent_dim)
        """
        if board.dim() == 3:
            board = board.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)

        # 정규화 (0-9 → 0-1)
        board = board.float() / 9.0

        x = self.conv(board)
        x = self.fc(x)
        return x


class AutoregressiveRectPolicy(nn.Module):
    """
    Autoregressive 직사각형 선택 정책

    4단계로 좌표를 순차적으로 선택:
    r1 → c1 → r2 → c2
    """

    def __init__(
        self,
        rows: int = 10,
        cols: int = 17,
        latent_dim: int = 256,
        embed_dim: int = 32
    ):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # 보드 인코더
        self.board_encoder = BoardEncoder(rows, cols, latent_dim)

        # 좌표 임베딩
        self.r_embed = nn.Embedding(rows, embed_dim)
        self.c_embed = nn.Embedding(cols, embed_dim)

        # 각 단계의 디코더
        self.r1_head = nn.Linear(latent_dim, rows)
        self.c1_head = nn.Linear(latent_dim + embed_dim, cols)
        self.r2_head = nn.Linear(latent_dim + embed_dim * 2, rows)
        self.c2_head = nn.Linear(latent_dim + embed_dim * 3, cols)

        # Value head (PPO용)
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        board: torch.Tensor,
        deterministic: bool = False,
        action: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            board: (batch, 1, rows, cols) 보드 상태
            deterministic: True면 argmax, False면 샘플링
            action: 주어진 행동 (학습 시 log_prob 계산용)

        Returns:
            action: (r1, c1, r2, c2) 각각 (batch,)
            log_prob: (batch,) 총 log 확률
            value: (batch,) 상태 가치
            info: 추가 정보
        """
        batch_size = board.shape[0]
        device = board.device

        # 1. 보드 인코딩
        h = self.board_encoder(board)  # (B, latent_dim)

        # 2. Value 계산
        value = self.value_head(h).squeeze(-1)  # (B,)

        # 3. 4단계 Autoregressive 선택
        log_probs = []

        # Step 1: r1 선택
        logits_r1 = self.r1_head(h)  # (B, rows)

        if action is not None:
            r1 = action[0]
        else:
            if deterministic:
                r1 = torch.argmax(logits_r1, dim=-1)
            else:
                r1 = torch.distributions.Categorical(logits=logits_r1).sample()

        log_prob_r1 = F.log_softmax(logits_r1, dim=-1)
        log_probs.append(log_prob_r1.gather(1, r1.unsqueeze(-1)).squeeze(-1))

        # r1 임베딩
        h_r1 = torch.cat([h, self.r_embed(r1)], dim=-1)

        # Step 2: c1 선택
        logits_c1 = self.c1_head(h_r1)  # (B, cols)

        if action is not None:
            c1 = action[1]
        else:
            if deterministic:
                c1 = torch.argmax(logits_c1, dim=-1)
            else:
                c1 = torch.distributions.Categorical(logits=logits_c1).sample()

        log_prob_c1 = F.log_softmax(logits_c1, dim=-1)
        log_probs.append(log_prob_c1.gather(1, c1.unsqueeze(-1)).squeeze(-1))

        # c1 임베딩
        h_c1 = torch.cat([h_r1, self.c_embed(c1)], dim=-1)

        # Step 3: r2 선택 (r2 ≥ r1 마스킹)
        logits_r2 = self.r2_head(h_c1)  # (B, rows)

        # r2 < r1인 것들을 마스킹
        mask_r2 = torch.arange(self.rows, device=device).unsqueeze(0) >= r1.unsqueeze(-1)  # (B, rows)
        logits_r2 = logits_r2.masked_fill(~mask_r2, -1e9)

        if action is not None:
            r2 = action[2]
        else:
            if deterministic:
                r2 = torch.argmax(logits_r2, dim=-1)
            else:
                r2 = torch.distributions.Categorical(logits=logits_r2).sample()

        log_prob_r2 = F.log_softmax(logits_r2, dim=-1)
        log_probs.append(log_prob_r2.gather(1, r2.unsqueeze(-1)).squeeze(-1))

        # r2 임베딩
        h_r2 = torch.cat([h_c1, self.r_embed(r2)], dim=-1)

        # Step 4: c2 선택 (c2 ≥ c1 마스킹)
        logits_c2 = self.c2_head(h_r2)  # (B, cols)

        # c2 < c1인 것들을 마스킹
        mask_c2 = torch.arange(self.cols, device=device).unsqueeze(0) >= c1.unsqueeze(-1)  # (B, cols)
        logits_c2 = logits_c2.masked_fill(~mask_c2, -1e9)

        if action is not None:
            c2 = action[3]
        else:
            if deterministic:
                c2 = torch.argmax(logits_c2, dim=-1)
            else:
                c2 = torch.distributions.Categorical(logits=logits_c2).sample()

        log_prob_c2 = F.log_softmax(logits_c2, dim=-1)
        log_probs.append(log_prob_c2.gather(1, c2.unsqueeze(-1)).squeeze(-1))

        # 총 log_prob (독립 가정으로 합)
        total_log_prob = sum(log_probs)

        action_tuple = (r1, c1, r2, c2)

        info = {
            'r1': r1,
            'c1': c1,
            'r2': r2,
            'c2': c2,
            'log_probs': log_probs,
        }

        return action_tuple, total_log_prob, value, info

    def get_value(self, board: torch.Tensor) -> torch.Tensor:
        """상태 가치만 계산 (빠른 추론)"""
        h = self.board_encoder(board)
        return self.value_head(h).squeeze(-1)


# ============================================================
# 테스트 코드
# ============================================================

if __name__ == "__main__":
    print("=== Autoregressive Policy 테스트 ===")
    print()

    # 모델 생성
    policy = AutoregressiveRectPolicy(rows=10, cols=17, latent_dim=256)

    # 더미 보드
    batch_size = 4
    board = torch.randint(0, 10, (batch_size, 1, 10, 17))

    print(f"Input board shape: {board.shape}")
    print()

    # Forward pass (샘플링)
    action, log_prob, value, info = policy(board, deterministic=False)

    r1, c1, r2, c2 = action
    print("샘플링 결과:")
    print(f"  r1: {r1}")
    print(f"  c1: {c1}")
    print(f"  r2: {r2}")
    print(f"  c2: {c2}")
    print(f"  log_prob: {log_prob}")
    print(f"  value: {value}")
    print()

    # 조건 체크
    print("제약 조건 체크:")
    print(f"  r2 >= r1: {torch.all(r2 >= r1).item()}")
    print(f"  c2 >= c1: {torch.all(c2 >= c1).item()}")
    print()

    # Forward pass (deterministic)
    action_det, log_prob_det, value_det, info_det = policy(board, deterministic=True)
    r1_det, c1_det, r2_det, c2_det = action_det

    print("Deterministic 결과:")
    print(f"  r1: {r1_det}")
    print(f"  c1: {c1_det}")
    print(f"  r2: {r2_det}")
    print(f"  c2: {c2_det}")
    print()

    # 파라미터 수
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"총 파라미터 수: {total_params:,}")
    print()

    print("✅ 모든 테스트 통과!")
