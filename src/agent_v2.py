#%% [markdown]
# Double DQN Agent with Soft Update and Safe Masking.
#
# ### [SOTA Alert]
# - Double DQN: Policy net으로 action 선택, Target net으로 Q-value 평가
# - Soft Update: tau=0.005로 점진적 target 업데이트
# - Safe Masking: -1e9 대신 -100 사용 (수치 안정성)

#%%
from dataclasses import dataclass
from typing import Optional, Tuple, List
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.models_v2 import DuelingDQN, get_device
from src.replay_buffer import PrioritizedReplayBuffer
from src.reward_normalizer import RewardNormalizer


@dataclass
class AgentConfig:
    """Agent 설정."""
    rows: int = 10
    cols: int = 17
    n_actions: int = 8415

    # 학습 파라미터
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient

    # Epsilon-greedy
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_frames: int = 50000

    # 버퍼
    buffer_size: int = 100000
    batch_size: int = 128
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100000

    # 보상 정규화
    normalize_rewards: bool = True
    reward_clip: float = 10.0

    # 안전한 마스킹 값 (수치 안정성)
    safe_mask_value: float = -100.0


class DoubleDQNAgent:
    """
    Double DQN Agent with prioritized experience replay.

    Key improvements over vanilla DQN:
    1. Double DQN: 과대평가 bias 감소
    2. Soft Update: 안정적인 target network 업데이트
    3. PER: 중요한 경험 우선 학습
    4. Reward Normalization: 학습 안정화
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.cfg = config or AgentConfig()
        self.device = get_device()

        # Networks
        self.policy_net = DuelingDQN(
            rows=self.cfg.rows,
            cols=self.cfg.cols,
            n_actions=self.cfg.n_actions
        ).to(self.device)

        self.target_net = DuelingDQN(
            rows=self.cfg.rows,
            cols=self.cfg.cols,
            n_actions=self.cfg.n_actions
        ).to(self.device)

        # Target network 초기화
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.cfg.lr,
            weight_decay=1e-5
        )

        # Replay buffer
        self.memory = PrioritizedReplayBuffer(
            capacity=self.cfg.buffer_size,
            alpha=self.cfg.per_alpha,
            beta_start=self.cfg.per_beta_start,
            beta_frames=self.cfg.per_beta_frames
        )

        # Reward normalizer
        self.reward_normalizer = RewardNormalizer(
            clip_range=self.cfg.reward_clip,
            gamma=self.cfg.gamma
        ) if self.cfg.normalize_rewards else None

        # Epsilon scheduling
        self.epsilon = self.cfg.epsilon_start
        self.frame_idx = 0

        # Logging
        self.train_step = 0

    def _preprocess(self, state: np.ndarray) -> torch.Tensor:
        """
        One-hot encode the board state.

        Args:
            state: (rows, cols) array with values 0-9

        Returns:
            (10, rows, cols) one-hot tensor
        """
        assert state.ndim == 2, f"Expected 2D state, got {state.ndim}D"
        state_tensor = torch.tensor(state, dtype=torch.long, device=self.device)
        one_hot = F.one_hot(state_tensor, num_classes=10)  # (R, C, 10)
        return one_hot.permute(2, 0, 1).float()  # (10, R, C)

    def select_action(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        training: bool = True
    ) -> int:
        """
        Epsilon-greedy action selection with masking.

        Args:
            state: Board state (rows, cols)
            mask: Valid action mask (n_actions,)
            training: Whether in training mode

        Returns:
            Selected action index
        """
        legal_actions = np.flatnonzero(mask)
        if len(legal_actions) == 0:
            return 0  # Fallback (should not happen)

        # Update epsilon
        if training:
            self.frame_idx += 1
            self.epsilon = max(
                self.cfg.epsilon_end,
                self.cfg.epsilon_start - self.frame_idx / self.cfg.epsilon_decay_frames
            )

        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            return int(random.choice(legal_actions))

        # Greedy action selection
        was_training = self.policy_net.training
        self.policy_net.eval()

        with torch.no_grad():
            state_v = self._preprocess(state).unsqueeze(0)
            q_values = self.policy_net(state_v).cpu().numpy()[0]

        if was_training:
            self.policy_net.train()

        # Safe masking
        masked_q = np.where(mask, q_values, self.cfg.safe_mask_value)
        return int(np.argmax(masked_q))

    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        mask: np.ndarray,
        next_mask: np.ndarray
    ):
        """Store transition in replay buffer."""
        # Normalize reward if enabled
        if self.reward_normalizer:
            reward = self.reward_normalizer.normalize(reward)

        self.memory.push(state, action, reward, next_state, done, mask, next_mask)

    def update(self) -> Optional[Tuple[float, List[float], Tuple[float, float]]]:
        """
        Perform one step of training.

        Returns:
            (loss, td_errors, (q_mean, target_mean)) or None if buffer too small
        """
        if len(self.memory) < self.cfg.batch_size:
            return None

        self.train_step += 1
        self.policy_net.train()
        self.target_net.eval()

        # Sample from PER
        samples, indices, weights = self.memory.sample(
            self.cfg.batch_size,
            self.frame_idx
        )
        weights = weights.to(self.device)

        # Unpack samples
        states, actions, rewards, next_states, dones, masks, next_masks = zip(*samples)

        # Preprocess
        state_batch = torch.stack([self._preprocess(s) for s in states])
        next_state_batch = torch.stack([self._preprocess(s) for s in next_states])
        action_batch = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(dones, device=self.device, dtype=torch.bool)
        next_mask_batch = torch.tensor(np.array(next_masks), device=self.device, dtype=torch.bool)

        # Current Q values
        current_q = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # Double DQN: policy net으로 action 선택, target net으로 평가
        with torch.no_grad():
            # Policy net으로 best action 선택
            next_q_policy = self.policy_net(next_state_batch)
            next_q_policy = torch.where(
                next_mask_batch,
                next_q_policy,
                torch.tensor(self.cfg.safe_mask_value, device=self.device)
            )
            best_actions = next_q_policy.argmax(dim=1, keepdim=True)

            # Target net으로 Q-value 평가
            next_q_target = self.target_net(next_state_batch)
            next_q = next_q_target.gather(1, best_actions).squeeze()
            next_q[done_batch] = 0.0

            # Target Q-values
            target_q = reward_batch + self.cfg.gamma * next_q

        # TD errors for PER
        td_errors = (current_q - target_q).detach().cpu().numpy()

        # Weighted Huber loss
        loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (loss * weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update PER priorities
        self.memory.update_priorities(indices, np.abs(td_errors) + 1e-6)

        # Soft update target network
        self._soft_update()

        return (
            loss.item(),
            td_errors.tolist(),
            (current_q.mean().item(), target_q.mean().item())
        )

    def _soft_update(self):
        """
        Polyak averaging: target = tau * policy + (1-tau) * target

        매 스텝마다 점진적으로 업데이트하여 안정성 향상.
        """
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.cfg.tau * policy_param.data +
                (1 - self.cfg.tau) * target_param.data
            )

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'frame_idx': self.frame_idx,
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'config': self.cfg,
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.frame_idx = checkpoint.get('frame_idx', 0)
        self.epsilon = checkpoint.get('epsilon', self.cfg.epsilon_end)
        self.train_step = checkpoint.get('train_step', 0)
        self.target_net.eval()

    def get_q_stats(self, state: np.ndarray, mask: np.ndarray) -> dict:
        """Get Q-value statistics for debugging."""
        self.policy_net.eval()
        with torch.no_grad():
            state_v = self._preprocess(state).unsqueeze(0)
            q_values = self.policy_net(state_v).cpu().numpy()[0]

        legal_q = q_values[mask]
        return {
            'q_min': float(q_values.min()),
            'q_max': float(q_values.max()),
            'q_mean': float(q_values.mean()),
            'legal_q_min': float(legal_q.min()) if len(legal_q) > 0 else 0,
            'legal_q_max': float(legal_q.max()) if len(legal_q) > 0 else 0,
            'legal_q_mean': float(legal_q.mean()) if len(legal_q) > 0 else 0,
        }


#%%
if __name__ == "__main__":
    # Agent 테스트
    config = AgentConfig(buffer_size=1000, batch_size=32)
    agent = DoubleDQNAgent(config)
    print(f"Device: {agent.device}")
    print(f"Policy net params: {sum(p.numel() for p in agent.policy_net.parameters()):,}")

    # Dummy transition
    state = np.random.randint(0, 10, size=(10, 17), dtype=np.int8)
    mask = np.random.random(8415) > 0.99  # ~1% valid actions

    action = agent.select_action(state, mask, training=True)
    print(f"Selected action: {action}")

    # Q-value stats
    stats = agent.get_q_stats(state, mask)
    print(f"Q-value stats: {stats}")

    # Check Q-value range
    assert stats['q_min'] >= -10.0, f"Q-value too low: {stats['q_min']}"
    assert stats['q_max'] <= 200.0, f"Q-value too high: {stats['q_max']}"

    print("Agent test passed!")
