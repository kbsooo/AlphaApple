#%% [markdown]
# Reward Normalization utilities.
#
# ### [SOTA Alert]
# - Running Mean/Std (Welford's algorithm): 온라인 통계 계산
# - Reward Clipping: 극단적인 보상 제한
# - Stable Baselines3 스타일 구현

#%%
from typing import Optional, Union
import numpy as np


class RunningMeanStd:
    """
    Online Mean/Std calculation using Welford's algorithm.

    수치적으로 안정적인 방식으로 평균과 분산을 점진적으로 업데이트.
    """

    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: Union[np.ndarray, float]):
        """Update running statistics with new data."""
        if isinstance(x, (int, float)):
            x = np.array([x])
        else:
            x = np.asarray(x)

        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.size

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: float,
        batch_var: float,
        batch_count: int
    ):
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.var = M2 / total_count
        self.count = total_count

    @property
    def std(self) -> float:
        """Standard deviation."""
        return np.sqrt(self.var)


class RewardNormalizer:
    """
    Reward Normalizer with clipping.

    - Return-based normalization (discounted return의 std로 나눔)
    - Clipping으로 극단값 제한
    """

    def __init__(
        self,
        clip_range: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8
    ):
        self.clip_range = clip_range
        self.gamma = gamma
        self.epsilon = epsilon

        self.rms = RunningMeanStd()
        self.returns = 0.0

    def normalize(self, reward: float, update: bool = True) -> float:
        """
        Normalize reward.

        Discounted return을 추적하고 그 분산으로 정규화.
        PPO/SAC 등에서 사용하는 방식.

        Args:
            reward: Raw reward
            update: Whether to update running statistics

        Returns:
            Normalized and clipped reward
        """
        # Discounted return 업데이트
        self.returns = self.returns * self.gamma + reward

        if update:
            self.rms.update(self.returns)

        # 표준편차로 정규화
        normalized = reward / (self.rms.std + self.epsilon)

        # Clipping
        return float(np.clip(normalized, -self.clip_range, self.clip_range))

    def reset(self):
        """Reset episode return (에피소드 경계에서 호출)."""
        self.returns = 0.0


class RewardScaler:
    """
    Simple reward scaling without normalization.

    정규화 대신 고정 스케일 적용.
    """

    def __init__(self, scale: float = 0.01, clip_range: Optional[float] = None):
        self.scale = scale
        self.clip_range = clip_range

    def normalize(self, reward: float, update: bool = True) -> float:
        """Scale reward."""
        scaled = reward * self.scale
        if self.clip_range is not None:
            scaled = np.clip(scaled, -self.clip_range, self.clip_range)
        return float(scaled)


#%%
if __name__ == "__main__":
    # RunningMeanStd 테스트
    rms = RunningMeanStd()
    for i in range(1000):
        rms.update(np.random.randn(10) * 5 + 10)
    print(f"Mean: {rms.mean:.2f} (expected ~10)")
    print(f"Std: {rms.std:.2f} (expected ~5)")

    # RewardNormalizer 테스트
    normalizer = RewardNormalizer(clip_range=10.0, gamma=0.99)

    rewards = []
    normalized_rewards = []
    for _ in range(100):
        r = np.random.random() * 10  # 0-10 범위
        nr = normalizer.normalize(r)
        rewards.append(r)
        normalized_rewards.append(nr)

    print(f"\nRaw rewards: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}")
    print(f"Normalized: mean={np.mean(normalized_rewards):.2f}, std={np.std(normalized_rewards):.2f}")
    print(f"Normalized range: [{min(normalized_rewards):.2f}, {max(normalized_rewards):.2f}]")

    # Clipping 확인
    assert all(-10 <= r <= 10 for r in normalized_rewards), "Clipping failed"

    print("\nReward normalizer test passed!")
