#%% [markdown]
# Prioritized Experience Replay (PER) implementation.
#
# ### [SOTA Alert]
# - TD-error 기반 우선순위 샘플링으로 중요한 경험 우선 학습
# - Importance Sampling으로 bias 보정
# - Sum Tree로 O(log n) 샘플링

#%%
from typing import List, Tuple, Any, Optional
import numpy as np
import torch


class SumTree:
    """
    Sum Tree for efficient priority-based sampling.

    - 리프 노드: 각 transition의 priority
    - 내부 노드: 자식들의 합
    - 루트: 전체 priority 합

    O(log n)으로 priority 기반 샘플링 가능.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Update parent nodes."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf index for cumulative sum s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Total priority sum."""
        return self.tree[0]

    def add(self, priority: float, data: Any):
        """Add new data with priority."""
        idx = self.write_idx + self.capacity - 1

        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        """Update priority at index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get data for cumulative sum s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def __len__(self) -> int:
        return self.n_entries


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.

    P(i) = p_i^alpha / sum(p^alpha)
    w_i = (N * P(i))^(-beta)

    - alpha: prioritization exponent (0=uniform, 1=full prioritization)
    - beta: importance sampling exponent (anneals to 1)
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon

        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def push(self, *transition):
        """Add transition with max priority."""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(
        self,
        batch_size: int,
        frame_idx: int
    ) -> Tuple[List[Any], List[int], torch.Tensor]:
        """
        Sample batch with importance sampling weights.

        Args:
            batch_size: Number of samples
            frame_idx: Current frame for beta annealing

        Returns:
            (samples, indices, weights)
        """
        # Beta annealing
        beta = min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

        samples = []
        indices = []
        priorities = []

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            samples.append(data)

        # Importance sampling weights
        probs = np.array(priorities) / self.tree.total()
        weights = (len(self.tree) * probs) ** (-beta)
        weights = weights / weights.max()  # Normalize

        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.tree)


class SimpleReplayBuffer:
    """
    Simple uniform replay buffer (fallback).

    PER보다 빠르지만 샘플 효율성 낮음.
    """

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position = 0

    def push(self, *transition):
        """Add transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


#%%
if __name__ == "__main__":
    # PER 테스트
    buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta_start=0.4)

    # Dummy transitions 추가
    for i in range(100):
        state = np.random.randint(0, 10, size=(10, 17), dtype=np.int8)
        action = np.random.randint(0, 8415)
        reward = np.random.random()
        next_state = np.random.randint(0, 10, size=(10, 17), dtype=np.int8)
        done = np.random.random() > 0.9
        mask = np.random.random(8415) > 0.99
        next_mask = np.random.random(8415) > 0.99

        buffer.push(state, action, reward, next_state, done, mask, next_mask)

    print(f"Buffer size: {len(buffer)}")

    # Sample
    samples, indices, weights = buffer.sample(batch_size=32, frame_idx=1000)
    print(f"Samples: {len(samples)}")
    print(f"Indices: {len(indices)}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Update priorities
    td_errors = np.random.random(32)
    buffer.update_priorities(indices, td_errors)

    print("PER test passed!")
