import numpy as np
from typing import Tuple, List, Optional

class BackwardBoardGenerator:
    """
    Generates a FruitBox board by starting from an empty board and 
    backwards filling rectangles that sum to 10.
    """
    def __init__(self, rows: int = 10, cols: int = 17, seed: Optional[int] = None):
        self.rows = rows
        self.cols = cols
        self.np_random = np.random.default_rng(seed)
        
        # Precompute all possible rectangles
        self.rects = []
        for r1 in range(rows):
            for r2 in range(r1, rows):
                for c1 in range(cols):
                    for c2 in range(c1, cols):
                        self.rects.append((r1, c1, r2, c2))
        self.rects = np.array(self.rects)

    def generate(self, target_coverage: float = 0.9) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Generates a board and a solution sequence.
        
        Args:
            target_coverage: Ratio of cells that should be part of the backward filling.
        
        Returns:
            board: (rows, cols) array of integers 1-9.
            solution: List of rectangles (r1, c1, r2, c2) in reverse order of generation.
        """
        board = np.zeros((self.rows, self.cols), dtype=np.int16)
        solution = []
        total_cells = self.rows * self.cols
        target_filled = int(total_cells * target_coverage)
        
        filled_count = 0
        attempts = 0
        max_attempts = 10000
        
        while filled_count < target_filled and attempts < max_attempts:
            attempts += 1
            # Sample a random rectangle
            idx = self.np_random.integers(0, len(self.rects))
            r1, c1, r2, c2 = self.rects[idx]
            
            region = board[r1:r2+1, c1:c2+1]
            empty_indices = np.where(region == 0)
            n_empty = len(empty_indices[0])
            
            if n_empty < 2:
                continue
            
            # Decide how many cells to fill (k between 2 and n_empty, capped at 10)
            k = self.np_random.integers(2, min(n_empty, 10) + 1)
            
            # Partition 10 into k positive integers
            # Standard "stars and bars" approach to get k positive integers summing to 10:
            # Pick k-1 divider positions from 9 possible slots (between 10 stars)
            dividers = sorted(self.np_random.choice(range(1, 10), k-1, replace=False))
            values = []
            prev = 0
            for d in dividers:
                values.append(d - prev)
                prev = d
            values.append(10 - prev)
            
            # Assign values to random empty cells in the region
            perm = self.np_random.permutation(n_empty)
            for i in range(k):
                rr = empty_indices[0][perm[i]]
                cc = empty_indices[1][perm[i]]
                board[r1 + rr, c1 + cc] = values[i]
            
            filled_count += k
            solution.append((r1, c1, r2, c2))
            
        # Fill remaining zeros with random numbers 1-9 to ensure no zeros
        remaining_zeros = np.where(board == 0)
        for r, c in zip(remaining_zeros[0], remaining_zeros[1]):
            board[r, c] = self.np_random.integers(1, 10)
            
        # The solution is in order of creation, so the agent should take them in REVERSE order
        # to guarantee solvability in the backward-filled parts.
        return board, solution[::-1]

if __name__ == "__main__":
    gen = BackwardBoardGenerator(rows=10, cols=17, seed=42)
    board, sol = gen.generate(target_coverage=0.95)
    print("Board sum:", board.sum())
    print("Zeros count:", np.sum(board == 0))
    print("Solution length:", len(sol))
    print("First 5 steps of solution:", sol[:5])
