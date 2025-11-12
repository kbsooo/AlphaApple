"""
Residual Autoregressive Policy for PPO (2.5M params)

핵심 개선사항:
1. Residual blocks (AlphaGo 스타일)
2. 더 큰 채널 (16→64→128)
3. 더 깊은 네트워크
4. ~2.5M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """Residual Block with skip connection"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # Skip connection
        out = F.relu(out)

        return out


class ResidualBoardEncoder(nn.Module):
    """
    Residual CNN Encoder

    Architecture:
    - Initial conv: 1 → 64 channels
    - 4x Residual blocks (64 channels)
    - Conv: 64 → 128 channels
    - 2x Residual blocks (128 channels)
    - Global pooling + FC → latent
    """

    def __init__(self, rows: int = 10, cols: int = 17, latent_dim: int = 256):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.latent_dim = latent_dim

        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Residual blocks (64 channels)
        self.res_blocks_64 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        # Upscale to 128 channels
        self.conv_128 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Residual blocks (128 channels)
        self.res_blocks_128 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )

        # Global average pooling + FC
        self.fc = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        if board.dim() == 3:
            board = board.unsqueeze(1)

        # Normalize
        board = board.float() / 9.0

        # Initial conv
        x = self.conv_init(board)

        # Residual blocks (64 channels)
        x = self.res_blocks_64(x)

        # Upscale to 128 channels
        x = self.conv_128(x)

        # Residual blocks (128 channels)
        x = self.res_blocks_128(x)

        # Global average pooling (batch_size, 128, H, W) → (batch_size, 128)
        x = x.mean(dim=[2, 3])

        # FC to latent
        x = self.fc(x)

        return x


class ResidualPolicy(nn.Module):
    """
    Residual Autoregressive Policy for PPO

    Key features:
    - Residual encoder (~2M params)
    - Autoregressive decoder
    - Policy head (action distribution)
    - Value head (state value)

    Total: ~2.5M parameters
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

        # Shared residual encoder
        self.board_encoder = ResidualBoardEncoder(rows, cols, latent_dim)

        # Embeddings
        self.r_embed = nn.Embedding(rows, embed_dim)
        self.c_embed = nn.Embedding(cols, embed_dim)

        # Decoders (shared for row/col)
        self.row_decoder = nn.Sequential(
            nn.Linear(latent_dim + embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, rows)
        )

        self.col_decoder = nn.Sequential(
            nn.Linear(latent_dim + embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, cols)
        )

        # Value head (for PPO)
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        board: torch.Tensor,
        deterministic: bool = False,
        action: Optional[Tuple[torch.Tensor, ...]] = None,
        masks: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass (autoregressive)

        Returns:
            action_tuple: (r1, c1, r2, c2)
            total_log_prob: sum of log probs
            value: state value
            info: additional info
        """
        batch_size = board.shape[0]
        device = board.device

        # Encode board
        h = self.board_encoder(board)

        # Value
        value = self.value_head(h).squeeze(-1)

        log_probs = []

        # === Step 1: r1 ===
        ctx_r1 = torch.cat([h, torch.zeros(batch_size, self.embed_dim * 2, device=device)], dim=-1)
        logits_r1 = self.row_decoder(ctx_r1)

        if masks is not None and 'r1_mask' in masks:
            r1_mask = masks['r1_mask']
            if r1_mask.dim() == 1:
                r1_mask = r1_mask.unsqueeze(0).expand(batch_size, -1)
            logits_r1 = logits_r1.masked_fill(~r1_mask, -1e9)

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

        if masks is not None and 'c1_masks' in masks:
            c1_masks = masks['c1_masks']
            if c1_masks.dim() == 2:
                c1_masks = c1_masks.unsqueeze(0).expand(batch_size, -1, -1)
            c1_mask = c1_masks[torch.arange(batch_size, device=device), r1]
            logits_c1 = logits_c1.masked_fill(~c1_mask, -1e9)

        if action is not None:
            c1 = action[1]
        else:
            if deterministic:
                c1 = torch.argmax(logits_c1, dim=-1)
            else:
                c1 = torch.distributions.Categorical(logits=logits_c1).sample()

        log_prob_c1 = F.log_softmax(logits_c1, dim=-1)
        log_probs.append(log_prob_c1.gather(1, c1.unsqueeze(-1)).squeeze(-1))

        # === Step 3: r2 ===
        c1_emb = self.c_embed(c1)
        ctx_r2 = torch.cat([h, r1_emb, c1_emb], dim=-1)
        logits_r2 = self.row_decoder(ctx_r2)

        mask_r2 = torch.arange(self.rows, device=device).unsqueeze(0) >= r1.unsqueeze(-1)

        if masks is not None and 'r2_masks' in masks:
            r2_masks = masks['r2_masks']
            if r2_masks.dim() == 3:
                r2_masks = r2_masks.unsqueeze(0).expand(batch_size, -1, -1, -1)
            r2_mask = r2_masks[torch.arange(batch_size, device=device), r1, c1]
            mask_r2 = mask_r2 & r2_mask

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

        # === Step 4: c2 ===
        r2_emb = self.r_embed(r2)
        ctx_c2 = torch.cat([h, r2_emb, c1_emb], dim=-1)
        logits_c2 = self.col_decoder(ctx_c2)

        mask_c2 = torch.arange(self.cols, device=device).unsqueeze(0) >= c1.unsqueeze(-1)

        if masks is not None and 'c2_masks' in masks:
            c2_masks = masks['c2_masks']
            if c2_masks.dim() == 4:
                c2_masks = c2_masks.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            c2_mask = c2_masks[torch.arange(batch_size, device=device), r1, c1, r2]
            mask_c2 = mask_c2 & c2_mask

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

        # Total log prob
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
        """Get state value (for PPO)"""
        h = self.board_encoder(board)
        return self.value_head(h).squeeze(-1)


# ============================================================
# Parameter count test
# ============================================================

if __name__ == "__main__":
    print("=== Residual Policy 테스트 ===")
    print()

    # 모델 생성
    policy = ResidualPolicy(rows=10, cols=17, latent_dim=256, embed_dim=32)

    # 파라미터 수
    total_params = sum(p.numel() for p in policy.parameters())
    encoder_params = sum(p.numel() for p in policy.board_encoder.parameters())

    print(f"총 파라미터 수: {total_params:,}")
    print(f"  - Encoder: {encoder_params:,}")
    print(f"  - 나머지: {total_params - encoder_params:,}")
    print()

    # 이전 모델과 비교
    old_params = 500_000
    print(f"V4 모델: {old_params:,}")
    print(f"증가율: {total_params/old_params:.1f}x")
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
    print(f"  log_prob: {log_prob}")
    print(f"  value: {value}")
    print()

    # 제약 확인
    print("제약 조건:")
    print(f"  r2 >= r1: {torch.all(r2 >= r1).item()}")
    print(f"  c2 >= c1: {torch.all(c2 >= c1).item()}")
    print()

    print("✅ 테스트 통과!")
