#%% [markdown]
# Dueling DQN with Q-value stabilization for FruitBox.
#
# ### [SOTA Alert]
# - Dueling Architecture (Wang et al., 2016): V(s)와 A(s,a) 분리로 state 가치 학습 향상
# - Residual Blocks: Pre-activation 스타일로 gradient flow 개선
# - Q-value Clipping: 출력 범위 제한으로 폭발 방지

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Pre-activation Residual Block.

    BN -> ReLU -> Conv 순서로 gradient가 더 안정적으로 흐름.
    Skip connection으로 깊은 네트워크에서도 학습 가능.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection: 채널 수가 다르면 1x1 conv로 맞춤
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        # Pre-activation: BN -> ReLU -> Conv
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)

        return out + identity


class DuelingDQN(nn.Module):
    """
    Dueling DQN with Q-value stabilization.

    Q(s,a) = V(s) + A(s,a) - mean(A)

    - V(s): State value (이 상태가 얼마나 좋은가)
    - A(s,a): Advantage (이 행동이 평균보다 얼마나 좋은가)

    ### [Tensor Risk]
    Q-value를 [-10, 200] 범위로 제한하여 폭발 방지.
    FruitBox 최대 보상 ~170 + 보너스를 고려한 범위.
    """

    def __init__(self, rows: int = 10, cols: int = 17, n_actions: int = 8415):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.n_actions = n_actions

        # Q-value 범위 제한 (FruitBox 특성상 0~200 정도가 합리적)
        self.q_min = -10.0
        self.q_max = 200.0

        # Feature Extractor: Residual CNN
        # 채널 확장: 10 -> 64 -> 128 -> 128
        self.conv_input = nn.Conv2d(10, 64, kernel_size=3, padding=1)
        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(64, 128)
        self.res_block3 = ResidualBlock(128, 128)

        flat_size = 128 * rows * cols

        # Dueling Streams
        # Value Stream: 상태의 가치 (scalar)
        self.value_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage Stream: 각 행동의 상대적 이점
        self.advantage_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Xavier 초기화로 안정적인 학습 시작."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 10, rows, cols) - One-hot encoded board

        Returns:
            Q-values: (Batch, n_actions) - 범위 제한된 Q-values
        """
        assert x.dim() == 4, f"Expected 4D input, got {x.dim()}D"
        assert x.shape[1] == 10, f"Expected 10 channels, got {x.shape[1]}"
        assert x.shape[2] == self.rows and x.shape[3] == self.cols, \
            f"Expected ({self.rows}, {self.cols}), got ({x.shape[2]}, {x.shape[3]})"

        # Feature extraction
        features = F.relu(self.conv_input(x))
        features = self.res_block1(features)
        features = self.res_block2(features)
        features = self.res_block3(features)

        # Dueling aggregation
        value = self.value_stream(features)  # (B, 1)
        advantage = self.advantage_stream(features)  # (B, n_actions)

        # Q = V + (A - mean(A))
        # Advantage의 평균을 빼서 identifiability 보장
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # Q-value clipping으로 폭발 방지
        q_values = torch.clamp(q_values, self.q_min, self.q_max)

        return q_values

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """중간 feature 추출 (디버깅/시각화용)."""
        features = F.relu(self.conv_input(x))
        features = self.res_block1(features)
        features = self.res_block2(features)
        features = self.res_block3(features)
        return features


#%%
def get_device() -> torch.device:
    """사용 가능한 최적의 디바이스 반환."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


#%%
if __name__ == "__main__":
    # 모델 테스트
    device = get_device()
    print(f"Device: {device}")

    model = DuelingDQN(rows=10, cols=17, n_actions=8415).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass 테스트
    dummy_input = torch.randn(4, 10, 10, 17, device=device)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Q-value range: [{output.min().item():.2f}, {output.max().item():.2f}]")

    assert output.shape == (4, 8415), f"Unexpected output shape: {output.shape}"
    assert output.min() >= -10.0, f"Q-value below minimum: {output.min()}"
    assert output.max() <= 200.0, f"Q-value above maximum: {output.max()}"

    print("Model test passed!")
