"""
PPO (Proximal Policy Optimization) 알고리즘

핵심 아이디어:
1. Rollout 수집 (policy로 환경 플레이)
2. Advantage 계산 (GAE)
3. PPO loss로 policy 업데이트 (clipped objective)
4. Value function 업데이트 (MSE loss)

References:
- PPO paper: https://arxiv.org/abs/1707.06347
- Schulman et al. 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


class PPOBuffer:
    """
    Rollout buffer for PPO

    Stores trajectories and computes advantages using GAE
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Buffers
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        self.dones = []

        # Computed
        self.advantages = None
        self.returns = None

    def add(
        self,
        obs: np.ndarray,
        action: Tuple,
        reward: float,
        value: float,
        log_prob: float,
        mask: dict,
        done: bool
    ):
        """Add a single transition"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(mask)
        self.dones.append(done)

    def finish_path(self, last_value: float = 0.0):
        """
        Compute advantages and returns using GAE

        Called at the end of an episode or when buffer is full
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [True])

        # GAE computation
        advantages = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                gae = 0

            # TD residual
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]

            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Returns = advantages + values
        returns = advantages + values[:-1]

        self.advantages = advantages
        self.returns = returns

    def get(self) -> Dict:
        """Get all data as dict"""
        return {
            'observations': self.observations,
            'actions': self.actions,
            'old_log_probs': self.log_probs,
            'advantages': self.advantages,
            'returns': self.returns,
            'masks': self.masks,
        }

    def clear(self):
        """Clear buffer"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        self.dones = []
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.observations)


class PPOTrainer:
    """
    PPO Trainer

    Handles PPO updates with clipped objective
    """

    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda'
    ):
        self.policy = policy
        self.device = device

        # Hyperparameters
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Optimizer
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def compute_loss(
        self,
        obs_batch: torch.Tensor,
        action_batch: Tuple[torch.Tensor, ...],
        old_log_prob_batch: torch.Tensor,
        advantage_batch: torch.Tensor,
        return_batch: torch.Tensor,
        mask_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO loss

        Returns:
            loss: total loss
            info: dict with loss components
        """
        # Forward pass
        _, log_prob, value, _ = self.policy(
            obs_batch,
            action=action_batch,
            masks=mask_batch
        )

        # === Policy loss (PPO clipped objective) ===
        ratio = torch.exp(log_prob - old_log_prob_batch)
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)

        policy_loss_1 = advantage_batch * ratio
        policy_loss_2 = advantage_batch * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # === Value loss ===
        value_loss = F.mse_loss(value, return_batch)

        # === Entropy bonus (for exploration) ===
        # For discrete actions, entropy = -sum(p * log(p))
        # We approximate using log_prob (not exact but good enough)
        entropy = -log_prob.mean()

        # === Total loss ===
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # Compute clip fraction (for monitoring)
        with torch.no_grad():
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
            approx_kl = torch.mean(old_log_prob_batch - log_prob).item()

        info = {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'clip_fraction': clip_fraction,
            'approx_kl': approx_kl,
        }

        return loss, info

    def update(
        self,
        buffer: PPOBuffer,
        n_epochs: int = 4,
        batch_size: int = 64
    ) -> Dict:
        """
        Update policy using PPO

        Args:
            buffer: PPO buffer with rollout data
            n_epochs: number of epochs to train
            batch_size: mini-batch size

        Returns:
            info: dict with training statistics
        """
        # Get data from buffer
        data = buffer.get()

        observations = data['observations']
        actions = data['actions']
        old_log_probs = torch.tensor(data['old_log_probs'], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(data['advantages'], dtype=torch.float32, device=self.device)
        returns = torch.tensor(data['returns'], dtype=torch.float32, device=self.device)
        masks = data['masks']

        n_samples = len(observations)

        # Training statistics
        all_info = []

        # Multiple epochs of SGD
        for epoch in range(n_epochs):
            # Random permutation
            indices = np.random.permutation(n_samples)

            # Mini-batch updates
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                if end > n_samples:
                    end = n_samples

                mb_indices = indices[start:end]

                # Prepare mini-batch
                mb_obs = torch.stack([
                    torch.from_numpy(observations[i]).float()
                    for i in mb_indices
                ]).to(self.device)

                mb_actions = tuple(
                    torch.tensor([actions[i][j] for i in mb_indices], dtype=torch.long, device=self.device)
                    for j in range(4)  # (r1, c1, r2, c2)
                )

                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Prepare masks (need to batch them)
                mb_masks = {
                    'r1_mask': torch.stack([torch.from_numpy(masks[i]['r1_mask']) for i in mb_indices]).to(self.device),
                    'c1_masks': torch.stack([torch.from_numpy(masks[i]['c1_masks']) for i in mb_indices]).to(self.device),
                    'r2_masks': torch.stack([torch.from_numpy(masks[i]['r2_masks']) for i in mb_indices]).to(self.device),
                    'c2_masks': torch.stack([torch.from_numpy(masks[i]['c2_masks']) for i in mb_indices]).to(self.device),
                }

                # Compute loss
                loss, info = self.compute_loss(
                    mb_obs,
                    mb_actions,
                    mb_old_log_probs,
                    mb_advantages,
                    mb_returns,
                    mb_masks
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                all_info.append(info)

        # Average statistics
        avg_info = {
            key: np.mean([info[key] for info in all_info])
            for key in all_info[0].keys()
        }

        return avg_info


# ============================================================
# Rollout collection
# ============================================================

def collect_rollouts(
    policy: nn.Module,
    env_wrapper,
    buffer: PPOBuffer,
    n_steps: int = 2048,
    device: str = 'cuda',
    gamma: float = 0.99,
    seed_offset: int = 0
) -> Dict:
    """
    Collect rollouts using current policy

    Args:
        policy: current policy
        env_wrapper: environment wrapper (make_autoregressive_env)
        buffer: PPO buffer to store data
        n_steps: number of steps to collect
        device: device
        gamma: discount factor
        seed_offset: seed offset for different boards

    Returns:
        info: dict with rollout statistics
    """
    policy.eval()

    episode_rewards = []
    episode_lengths = []

    steps = 0
    episode_reward = 0
    episode_length = 0

    # Initialize environment
    from envs.backward_generator import BackwardBoardGenerator
    generator = BackwardBoardGenerator(rows=10, cols=17, seed=seed_offset)
    board, _ = generator.generate(target_coverage=0.95)

    env = env_wrapper.env
    env.board = board.astype(np.int16)
    obs = board.clip(0, 9).astype(np.int8)

    with torch.no_grad():
        while steps < n_steps:
            # Get masks
            masks_np = env_wrapper.get_autoregressive_masks()
            masks_torch = {
                'r1_mask': torch.from_numpy(masks_np['r1_mask']).to(device),
                'c1_masks': torch.from_numpy(masks_np['c1_masks']).to(device),
                'r2_masks': torch.from_numpy(masks_np['r2_masks']).to(device),
                'c2_masks': torch.from_numpy(masks_np['c2_masks']).to(device)
            }

            # Forward pass
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(device)
            action_tuple, log_prob, value, _ = policy(obs_tensor, deterministic=False, masks=masks_torch)

            # Extract action
            r1 = int(action_tuple[0][0].item())
            c1 = int(action_tuple[1][0].item())
            r2 = int(action_tuple[2][0].item())
            c2 = int(action_tuple[3][0].item())

            # Step environment
            next_obs, reward, terminated, truncated, info = env_wrapper.step_with_coords(r1, c1, r2, c2)

            # Store in buffer
            buffer.add(
                obs=obs.copy(),
                action=(r1, c1, r2, c2),
                reward=reward,
                value=value.item(),
                log_prob=log_prob.item(),
                mask=masks_np,
                done=terminated or truncated
            )

            episode_reward += reward
            episode_length += 1
            steps += 1

            obs = next_obs

            # Episode ended
            if terminated or truncated or episode_length >= 500:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                # Finish path
                if terminated or truncated:
                    buffer.finish_path(last_value=0.0)
                else:
                    # Bootstrap value
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(device)
                    last_value = policy.get_value(obs_tensor).item()
                    buffer.finish_path(last_value=last_value)

                # Reset environment
                seed_offset += 1
                generator = BackwardBoardGenerator(rows=10, cols=17, seed=seed_offset)
                board, _ = generator.generate(target_coverage=0.95)
                env.board = board.astype(np.int16)
                obs = board.clip(0, 9).astype(np.int8)

                episode_reward = 0
                episode_length = 0

    info = {
        'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0,
        'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0,
        'n_episodes': len(episode_rewards),
    }

    return info
