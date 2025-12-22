import torch
from src.models import FruitBoxDQN
rows, cols = 10, 17
n_actions = (rows*(rows+1)//2) * (cols*(cols+1)//2)
m = FruitBoxDQN(rows, cols, n_actions)
print(sum(p.numel() for p in m.parameters()))
print(sum(p.numel()*p.element_size() for p in m.parameters()) / 1024**2, "MiB")
