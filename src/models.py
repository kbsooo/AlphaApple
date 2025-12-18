import torch
import torch.nn as nn
import torch.nn.functional as F

class FruitBoxDQN(nn.Module):
    """
    CNN-based DQN model for FruitBox.
    Input: (Batch, 10, 10, 17) - One-hot encoded board (10 channels for 0-9)
    Output: Q-values for all possible actions.
    """
    def __init__(self, rows: int, cols: int, n_actions: int):
        super(FruitBoxDQN, self).__init__()
        self.rows = rows
        self.cols = cols
        
        # Encoder: Convolutional layers to process the grid
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Flattened size: 64 * rows * cols
        flat_features = 64 * rows * cols
        
        # Fully connected layers
        self.fc1 = nn.Linear(flat_features, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, 10, rows, cols)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
