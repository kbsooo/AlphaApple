"""
경량 Autoregressive Policy (500K params)

핵심 아이디어:
1. 더 작은 CNN (16 → 32 채널)
2. Shared encoder (r1,c1,r2,c2 모두 같은 인코더 사용)
3. 작은 embedding (16차원)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LightweightBoardEncoder(nn.Module):
    """경량 보드 인코더"""

    def __init__(self, rows: int = 10, cols: int = 17, latent_dim: int = 128):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.latent_dim = latent_dim

        # 더 작은 CNN
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = 32 * rows * cols

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, latent_dim),
            nn.ReLU(),
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        if board.dim() == 3:
            board = board.unsqueeze(1)

        board = board.float() / 9.0
        x = self.conv(board)
        x = self.fc(x)
        return x


class LightweightPolicy(nn.Module):
    """
    경량 Autoregressive Policy

    핵심 차이점:
    - Shared encoder (모든 단계가 같은 인코더 사용)
    - 작은 embedding (16차원)
    - 총 ~500K parameters
    """

    def __init__(
        self,
        rows: int = 10,
        cols: int = 17,
        latent_dim: int = 128,
        embed_dim: int = 16  # 32 → 16으로 축소
    ):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # Shared encoder
        self.board_encoder = LightweightBoardEncoder(rows, cols, latent_dim)

        # 작은 embedding
        self.r_embed = nn.Embedding(rows, embed_dim)
        self.c_embed = nn.Embedding(cols, embed_dim)

        # Shared decoder (모든 좌표에 동일 사용)
        self.row_decoder = nn.Linear(latent_dim + embed_dim * 2, rows)
        self.col_decoder = nn.Linear(latent_dim + embed_dim * 2, cols)

        # Value head
        self.value_head = nn.Linear(latent_dim, 1)

    def forward(
        self,
        board: torch.Tensor,
        deterministic: bool = False,
        action: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        batch_size = board.shape[0]
        device = board.device

        # 보드 인코딩 (한 번만)
        h = self.board_encoder(board)

        # Value
        value = self.value_head(h).squeeze(-1)

        log_probs = []

        # === Step 1: r1 ===
        # 초기 컨텍스트 (zero padding)
        ctx_r1 = torch.cat([h, torch.zeros(batch_size, self.embed_dim * 2, device=device)], dim=-1)
        logits_r1 = self.row_decoder(ctx_r1)

        if action is not None:
            r1 = action[0]
        else:
            if deterministic:
                r1 = torch.argmax(logits_r1, dim=-1)
            else:
                r1 = torch.distributions.Categorical(logits=logits_r1).sample()

        log_prob_r1 = F.log_softmax(logits_r1, dim=-1)
        log_probs.append(log_prob_r1.gather(1, r1.unsqueeze(-1)).squeeze(-1))

        # === Step 2: c1 ===
        r1_emb = self.r_embed(r1)
        ctx_c1 = torch.cat([h, r1_emb, torch.zeros(batch_size, self.embed_dim, device=device)], dim=-1)
        logits_c1 = self.col_decoder(ctx_c1)

        if action is not None:
            c1 = action[1]
        else:
            if deterministic:
                c1 = torch.argmax(logits_c1, dim=-1)
            else:
                c1 = torch.distributions.Categorical(logits=logits_c1).sample()

        log_prob_c1 = F.log_softmax(logits_c1, dim=-1)
        log_probs.append(log_prob_c1.gather(1, c1.unsqueeze(-1)).squeeze(-1))

        # === Step 3: r2 (r2 ≥ r1) ===
        c1_emb = self.c_embed(c1)
        ctx_r2 = torch.cat([h, r1_emb, c1_emb], dim=-1)
        logits_r2 = self.row_decoder(ctx_r2)

        # 마스킹
        mask_r2 = torch.arange(self.rows, device=device).unsqueeze(0) >= r1.unsqueeze(-1)
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

        # === Step 4: c2 (c2 ≥ c1) ===
        r2_emb = self.r_embed(r2)
        ctx_c2 = torch.cat([h, r2_emb, c1_emb], dim=-1)
        logits_c2 = self.col_decoder(ctx_c2)

        # 마스킹
        mask_c2 = torch.arange(self.cols, device=device).unsqueeze(0) >= c1.unsqueeze(-1)
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

        # 총 log_prob
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
        h = self.board_encoder(board)
        return self.value_head(h).squeeze(-1)


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    print("=== 경량 Policy 테스트 ===")
    print()

    # 모델 생성
    policy = LightweightPolicy(rows=10, cols=17, latent_dim=128)

    # 파라미터 수
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"총 파라미터 수: {total_params:,}")

    # 원래 모델과 비교
    original_params = 2_891_863
    print(f"원래 모델: {original_params:,}")
    print(f"축소율: {(1 - total_params/original_params)*100:.1f}%")
    print()

    # Forward test
    batch_size = 4
    board = torch.randint(0, 10, (batch_size, 1, 10, 17))

    action, log_prob, value, info = policy(board, deterministic=False)

    r1, c1, r2, c2 = action
    print("샘플링 결과:")
    print(f"  r1: {r1}")
    print(f"  c1: {c1}")
    print(f"  r2: {r2}")
    print(f"  c2: {c2}")
    print()

    # 제약 확인
    print("제약 조건:")
    print(f"  r2 >= r1: {torch.all(r2 >= r1).item()}")
    print(f"  c2 >= c1: {torch.all(c2 >= c1).item()}")
    print()

    print("✅ 테스트 통과!")
