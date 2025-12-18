import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.fruitbox_env_improved import FruitBoxEnvImproved, FruitBoxImprovedConfig

def test_backward_generation():
    print("Testing backward generation...")
    config = FruitBoxImprovedConfig(
        rows=10, 
        cols=17, 
        use_backward_generator=True, 
        target_coverage=0.9,
        render_mode="ansi"
    )
    env = FruitBoxEnvImproved(config=config)
    obs, info = env.reset(seed=42)
    
    print("Initial board sum:", obs.sum())
    print("Zero cells count:", np.sum(obs == 0))
    print("Initial legal actions:", len(info["action_mask"].nonzero()[0]))
    
    # Verify that the board is mostly non-zero
    assert np.mean(obs > 0) >= 0.8  # should be around 0.9 or more
    print("Backward generation test passed.")

def test_reward_structure():
    print("\nTesting reward structure...")
    config = FruitBoxImprovedConfig(
        rows=5, 
        cols=5, 
        reward_per_cell=1.0, 
        reward_per_zero_cell=0.0,
        use_backward_generator=False # Use random for easy manipulation
    )
    env = FruitBoxEnvImproved(config=config)
    env.reset(seed=42)
    
    # Force a controllable state
    env.board = np.zeros((5, 5), dtype=np.int16)
    env.board[0, 0] = 3
    env.board[0, 1] = 7
    env.board[0, 2] = 0 # This cell is zero
    
    # Re-calculate mask
    env._rect_sums, env._action_mask = env._compute_full_mask(env.board)
    
    # Action for [3, 7, 0] which is (0,0) to (0,2)
    # Find the action index for (0,0,0,2)
    action_idx = -1
    for i, rect in enumerate(env.rects):
        if np.array_equal(rect, [0, 0, 0, 2]):
            action_idx = i
            break
    
    assert action_idx != -1
    assert env._action_mask[action_idx] == True
    
    # Step with this action
    obs, reward, terminated, truncated, info = env.step(action_idx)
    
    print(f"Action [3, 7, 0] reward: {reward}")
    # Reward should be 2.0 (for 3 and 7), not 2.25 or anything else.
    assert reward == 2.0
    
    # Now test a dense one
    env.board[1, 0] = 2
    env.board[1, 1] = 1
    env.board[1, 2] = 4
    env.board[1, 3] = 3
    env._rect_sums, env._action_mask = env._compute_full_mask(env.board)
    
    action_idx_dense = -1
    for i, rect in enumerate(env.rects):
        if np.array_equal(rect, [1, 0, 1, 3]):
            action_idx_dense = i
            break
    
    obs, reward_dense, _, _, _ = env.step(action_idx_dense)
    print(f"Action [2, 1, 4, 3] reward: {reward_dense}")
    assert reward_dense == 4.0
    
    assert reward_dense > reward
    print("Reward structure test passed.")

if __name__ == "__main__":
    test_backward_generation()
    test_reward_structure()
