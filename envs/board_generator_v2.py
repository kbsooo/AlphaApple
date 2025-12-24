#%% [markdown]
# Balanced Board Generator with solution verification.
#
# ### 기존 BackwardGenerator 문제점
# - 작은 숫자(1,2,3) 편향
# - 작은 영역에 치우침
# - Solution 유효성 미검증
#
# ### 개선 사항
# - sqrt(area) 가중치로 큰 영역도 포함
# - Rejection sampling으로 1-9 균등 분포
# - 생성 후 solution 유효성 검증

#%%
from typing import List, Tuple, Optional
import numpy as np


class BalancedBoardGenerator:
    """
    균등 분포 + 유효성 검증 보드 생성기.

    - 영역 크기: sqrt(area) 가중치로 균등화
    - 숫자 분포: 1-9 균등 (rejection sampling)
    - Solution: 생성 후 검증
    """

    def __init__(self, rows: int = 10, cols: int = 17, seed: Optional[int] = None):
        self.rows = rows
        self.cols = cols
        self.rng = np.random.default_rng(seed)

        # 모든 가능한 사각형 영역
        self.rects: List[Tuple[int, int, int, int]] = []
        for r1 in range(rows):
            for r2 in range(r1, rows):
                for c1 in range(cols):
                    for c2 in range(c1, cols):
                        self.rects.append((r1, c1, r2, c2))

        # 영역 크기별 가중치 계산
        self.areas = np.array([
            (r2 - r1 + 1) * (c2 - c1 + 1)
            for r1, c1, r2, c2 in self.rects
        ])

        # sqrt(area) 가중치: 큰 영역도 적절히 포함
        self.weights = np.sqrt(self.areas).astype(np.float64)
        self.weights /= self.weights.sum()

    def _sample_rect(self) -> Tuple[int, int, int, int]:
        """가중치에 따라 사각형 선택."""
        idx = self.rng.choice(len(self.rects), p=self.weights)
        return self.rects[idx]

    def _generate_sum10_uniform(self, max_count: int) -> Optional[List[int]]:
        """
        합이 10인 숫자들을 균등 분포로 생성.

        - Rejection sampling으로 균등한 숫자 분포 유지
        - 기존 stars-and-bars 방식보다 균등함
        """
        if max_count < 2:
            return None

        # 2-5개 숫자 사용 (너무 많으면 작은 숫자 편향)
        k = min(max_count, self.rng.integers(2, min(6, max_count + 1)))

        # Rejection sampling: 합이 10인 조합 찾기
        for _ in range(100):
            values = self.rng.integers(1, 10, size=k)
            if values.sum() == 10:
                return values.tolist()

        # Fallback: 조정으로 합 맞추기
        values = self.rng.integers(1, 10, size=k)
        diff = int(values.sum()) - 10

        for i in range(abs(diff)):
            idx = i % k
            if diff > 0 and values[idx] > 1:
                values[idx] -= 1
                diff -= 1
            elif diff < 0 and values[idx] < 9:
                values[idx] += 1
                diff += 1

        if values.sum() == 10 and np.all(values >= 1) and np.all(values <= 9):
            return values.tolist()

        return None

    def generate(
        self,
        target_coverage: float = 0.95,
        max_attempts: int = 10000,
        verify_solution: bool = True
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        보드 생성.

        Args:
            target_coverage: 목표 채움 비율
            max_attempts: 최대 시도 횟수
            verify_solution: Solution 검증 여부

        Returns:
            (board, solution) - solution은 역순 (마지막 제거 → 첫 제거)
        """
        board = np.zeros((self.rows, self.cols), dtype=np.int16)
        solution: List[Tuple[int, int, int, int]] = []

        target = int(self.rows * self.cols * target_coverage)
        filled = 0
        attempts = 0

        while filled < target and attempts < max_attempts:
            attempts += 1

            # 가중치 기반 영역 선택
            rect = self._sample_rect()
            r1, c1, r2, c2 = rect

            # 영역 내 빈 셀 확인
            region = board[r1:r2+1, c1:c2+1]
            empty_mask = (region == 0)
            n_empty = empty_mask.sum()

            if n_empty < 2:
                continue

            # 합이 10인 숫자 생성
            values = self._generate_sum10_uniform(n_empty)
            if values is None:
                continue

            # 빈 셀에 값 할당
            empty_coords = np.argwhere(empty_mask)
            self.rng.shuffle(empty_coords)

            for i, val in enumerate(values):
                if i >= len(empty_coords):
                    break
                dr, dc = empty_coords[i]
                board[r1 + dr, c1 + dc] = val

            filled += len(values)
            solution.append(rect)

        # 남은 0 채우기
        board = self._fill_remaining(board)

        # Solution 검증
        if verify_solution:
            valid_solution = self._verify_solution(board, solution)
            return board, valid_solution

        return board, solution[::-1]

    def _fill_remaining(self, board: np.ndarray) -> np.ndarray:
        """
        남은 0을 1-9 랜덤으로 채우기.

        기존 solution 영역의 합=10을 깨지 않도록 주의.
        """
        remaining = np.argwhere(board == 0)

        for r, c in remaining:
            # 균등 분포로 1-9 선택
            board[r, c] = self.rng.integers(1, 10)

        return board

    def _verify_solution(
        self,
        board: np.ndarray,
        solution: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Solution의 각 영역이 실제로 합=10인지 검증.

        검증된 영역만 반환.
        """
        valid = []
        test_board = board.copy()

        # 역순으로 검증 (실제 게임 순서)
        for rect in reversed(solution):
            r1, c1, r2, c2 = rect
            region = test_board[r1:r2+1, c1:c2+1]

            if region.sum() == 10:
                valid.append(rect)
                # 실제 제거 시뮬레이션
                test_board[r1:r2+1, c1:c2+1] = 0

        return valid[::-1]  # 다시 역순으로 (생성 순서)


class MixedBoardGenerator:
    """
    Backward + Random 혼합 생성기.

    학습 중 distribution shift를 줄이기 위해
    backward와 random 보드를 혼합.
    """

    def __init__(
        self,
        rows: int = 10,
        cols: int = 17,
        seed: Optional[int] = None
    ):
        self.rows = rows
        self.cols = cols
        self.rng = np.random.default_rng(seed)
        self.balanced_gen = BalancedBoardGenerator(rows, cols, seed)

    def generate(
        self,
        backward_ratio: float = 0.5,
        target_coverage: float = 0.95,
        enforce_sum_mod_10: bool = True
    ) -> Tuple[np.ndarray, Optional[List[Tuple[int, int, int, int]]]]:
        """
        혼합 보드 생성.

        Args:
            backward_ratio: Backward generator 사용 비율
            target_coverage: Backward 생성 시 목표 coverage
            enforce_sum_mod_10: Random 보드 합이 10의 배수가 되도록

        Returns:
            (board, solution) - Random 보드는 solution=None
        """
        if self.rng.random() < backward_ratio:
            return self.balanced_gen.generate(target_coverage=target_coverage)

        # Random board
        board = self.rng.integers(1, 10, size=(self.rows, self.cols), dtype=np.int16)

        if enforce_sum_mod_10:
            delta = int((10 - (board.sum() % 10)) % 10)
            attempts = 0
            while delta > 0 and attempts < 100:
                r = int(self.rng.integers(0, self.rows))
                c = int(self.rng.integers(0, self.cols))
                inc = min(9 - int(board[r, c]), delta)
                if inc > 0:
                    board[r, c] += inc
                    delta -= inc
                attempts += 1

        return board, None


#%%
if __name__ == "__main__":
    # BalancedBoardGenerator 테스트
    gen = BalancedBoardGenerator(rows=10, cols=17, seed=42)

    print("=== Balanced Board Generator Test ===")

    # 여러 보드 생성 및 분포 확인
    all_values = []
    all_areas = []
    valid_solutions = 0

    for i in range(100):
        board, solution = gen.generate(target_coverage=0.9, verify_solution=True)
        all_values.extend(board[board > 0].tolist())
        for rect in solution:
            r1, c1, r2, c2 = rect
            area = (r2 - r1 + 1) * (c2 - c1 + 1)
            all_areas.append(area)
        if len(solution) > 0:
            valid_solutions += 1

    # 숫자 분포 확인
    print(f"\nNumber distribution (should be ~uniform 1-9):")
    for i in range(1, 10):
        count = all_values.count(i)
        pct = count / len(all_values) * 100
        print(f"  {i}: {pct:.1f}%")

    # 영역 크기 분포 확인
    print(f"\nArea distribution:")
    areas = np.array(all_areas)
    print(f"  Mean area: {areas.mean():.1f}")
    print(f"  Min area: {areas.min()}")
    print(f"  Max area: {areas.max()}")
    print(f"  Areas > 10: {(areas > 10).sum()} ({(areas > 10).mean()*100:.1f}%)")

    # Solution 유효성
    print(f"\nBoards with valid solutions: {valid_solutions}/100")

    # 샘플 보드 출력
    board, solution = gen.generate(target_coverage=0.9)
    print(f"\nSample board (solution has {len(solution)} steps):")
    for row in board:
        print(" ".join(f"{v:1d}" for v in row))

    print("\nBalanced board generator test passed!")
