import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from src.models import FruitBoxDQN, get_device

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))

    def sample(self, batch_size: int):
        samples = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mask, next_mask = zip(*samples)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done),
            np.array(mask),
            np.array(next_mask)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self, 
        rows: int, 
        cols: int, 
        n_actions: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 20000,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update: int = 1000
    ):
        self.rows = rows
        self.cols = cols
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = get_device()
        
        self.policy_net = FruitBoxDQN(rows, cols, n_actions).to(self.device)
        self.target_net = FruitBoxDQN(rows, cols, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.steps_done = 0

    def _preprocess(self, state: np.ndarray) -> torch.Tensor:
        """
        One-hot encode the board state (0-9).
        Input shape: (rows, cols)
        Output shape: (10, rows, cols)
        """
        state_tensor = torch.tensor(state, dtype=torch.long, device=self.device)
        one_hot = torch.nn.functional.one_hot(state_tensor, num_classes=10) # (R, C, 10)
        one_hot = one_hot.permute(2, 0, 1).float() # (10, R, C)
        return one_hot

    def select_action(self, state: np.ndarray, mask: np.ndarray, training: bool = True) -> int:
        self.steps_done += 1
        
        # Epsilon-greedy
        if training:
            self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay)
            
        if training and random.random() < self.epsilon:
            # Random valid action
            legal_indices = np.where(mask)[0]
            if len(legal_indices) == 0:
                return 0 # Should not happen if mask is correct
            return random.choice(legal_indices)
        else:
            with torch.no_grad():
                state_v = self._preprocess(state).unsqueeze(0) # (1, 10, R, C)
                q_values = self.policy_net(state_v).cpu().numpy()[0]
                
                # Apply mask: set illegal actions to a very low value
                masked_q_values = np.where(mask, q_values, -1e9)
                return int(np.argmax(masked_q_values))

    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones, masks, next_masks = self.memory.sample(self.batch_size)
        
        # Preprocess states
        state_batch = torch.stack([self._preprocess(s) for s in states])
        next_state_batch = torch.stack([self._preprocess(s) for s in next_states])
        action_batch = torch.tensor(actions, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.bool, device=self.device)
        next_mask_batch = torch.tensor(next_masks, device=self.device)

        # Q(s, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # max Q(s', a') with masking
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            # Apply next action mask
            next_q_values[~next_mask_batch] = -1e9
            next_state_values = next_q_values.max(1)[0]
            next_state_values[done_batch] = 0.0 # Done states have 0 future value
            
        # Expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()

    def save(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
