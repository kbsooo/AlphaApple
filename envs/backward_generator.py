"""
역방향 보드 생성기 (Backward Board Generator)

핵심 아이디어:
1. 빈 보드에서 시작
2. 합=10이 되는 블록들을 역으로 배치
3. 남은 공간은 랜덤으로 채움
4. 결과: target_coverage% 이상 제거 가능한 보드 보장
"""

import numpy as np
from typing import Tuple, List


class BackwardBoardGenerator:
    """역방향 보드 생성기"""

    def __init__(self, rows: int = 10, cols: int = 17, seed: int = None):
        self.rows = rows
        self.cols = cols
        self.rng = np.random.default_rng(seed)

    def generate(self, target_coverage: float = 0.7) -> Tuple[np.ndarray, List[Tuple]]:
        """
        풀 수 있는 보드를 역방향으로 생성

        Args:
            target_coverage: 제거 가능 셀 비율 (0.0-1.0)

        Returns:
            board: (rows, cols) 보드
            solution: 제거 가능한 블록들의 리스트 [(r1,c1,r2,c2), ...]
        """
        board = np.zeros((self.rows, self.cols), dtype=np.int8)
        solution = []  # 풀이 가능한 블록들

        target_cells = int(self.rows * self.cols * target_coverage)
        placed_cells = 0

        # Phase 1: 합=10 블록들을 배치
        attempts = 0
        max_attempts = 1000

        while placed_cells < target_cells and attempts < max_attempts:
            attempts += 1

            # 빈 영역에서 무작위 직사각형 선택
            rect = self._random_empty_rect(board)
            if rect is None:
                break

            r1, c1, r2, c2 = rect
            size = (r2 - r1 + 1) * (c2 - c1 + 1)

            # 합=10이 되도록 숫자 생성
            numbers = self._generate_numbers_sum_to_10(size)

            # 배치
            idx = 0
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    board[r, c] = numbers[idx]
                    idx += 1

            solution.append((r1, c1, r2, c2))
            placed_cells += size

        # Phase 2: 남은 빈 칸은 무작위로 채움
        empty_mask = (board == 0)
        num_empty = np.sum(empty_mask)
        if num_empty > 0:
            board[empty_mask] = self.rng.integers(1, 10, size=num_empty)

        return board, solution

    def _random_empty_rect(self, board: np.ndarray) -> Tuple[int, int, int, int] | None:
        """
        보드에서 완전히 비어있는 무작위 직사각형 찾기

        Returns:
            (r1, c1, r2, c2) 또는 None
        """
        # 빈 셀 찾기
        empty_cells = np.argwhere(board == 0)
        if len(empty_cells) == 0:
            return None

        # 무작위 시작점
        start_idx = self.rng.integers(0, len(empty_cells))
        r1, c1 = empty_cells[start_idx]

        # 크기 결정 (작은 크기 선호: 2-4 셀)
        max_size = min(4, len(empty_cells))  # 합=10 제약상 큰 블록 어려움
        target_size = self.rng.integers(2, max_size + 1)

        # 가능한 직사각형 형태들
        possible_shapes = []
        for h in range(1, min(target_size, self.rows - r1) + 1):
            for w in range(1, min(target_size, self.cols - c1) + 1):
                if h * w <= target_size:
                    possible_shapes.append((h, w))

        # 무작위로 형태 선택
        if not possible_shapes:
            return (r1, c1, r1, c1)  # 1×1

        h, w = possible_shapes[self.rng.integers(0, len(possible_shapes))]
        r2, c2 = r1 + h - 1, c1 + w - 1

        # 영역이 모두 비어있는지 확인
        if np.all(board[r1:r2+1, c1:c2+1] == 0):
            return (r1, c1, r2, c2)

        # 비어있지 않으면 축소
        for h_try in range(h, 0, -1):
            for w_try in range(w, 0, -1):
                r2_try = r1 + h_try - 1
                c2_try = c1 + w_try - 1
                if r2_try < self.rows and c2_try < self.cols:
                    if np.all(board[r1:r2_try+1, c1:c2_try+1] == 0):
                        return (r1, c1, r2_try, c2_try)

        return None

    def _generate_numbers_sum_to_10(self, size: int) -> List[int]:
        """
        size개의 숫자(1-9)를 생성하여 합=10

        전략:
        - size=2: (1,9), (2,8), (3,7), (4,6), (5,5) 중 선택
        - size=3: (1,1,8), (1,2,7), ..., (2,3,5), (3,3,4) 등
        - size=4: (1,1,1,7), (1,1,2,6), ..., (1,3,3,3), (2,2,2,4) 등
        """
        if size < 1 or size > 9:
            return [1] * size  # fallback

        # 간단한 분할 알고리즘
        numbers = []
        remaining = 10

        for i in range(size - 1):
            # 남은 칸에 최소 1씩 필요하므로
            remaining_slots = size - i - 1
            min_val = 1
            max_val = min(9, remaining - remaining_slots)

            if max_val < min_val:
                max_val = min_val

            val = int(self.rng.integers(min_val, max_val + 1))
            numbers.append(val)
            remaining -= val

        # 마지막 숫자
        last = remaining
        if last < 1:
            last = 1
        if last > 9:
            last = 9
        numbers.append(last)

        # 합이 10이 아니면 조정
        current_sum = sum(numbers)
        if current_sum != 10:
            # 첫 번째 숫자를 조정
            diff = 10 - current_sum
            numbers[0] = max(1, min(9, numbers[0] + diff))

        # 셔플 (패턴을 숨김)
        self.rng.shuffle(numbers)

        return numbers


# ============================================================
# 테스트 및 검증
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/user/AlphaApple')
    from envs.fruitbox_env import FruitBoxEnv, FruitBoxConfig

    print("=== 역방향 보드 생성기 테스트 ===")
    print()

    generator = BackwardBoardGenerator(rows=10, cols=17, seed=42)

    # 여러 목표 coverage 테스트
    for target_coverage in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"목표 Coverage: {target_coverage:.0%}")

        results = []
        for seed in range(10):
            gen = BackwardBoardGenerator(rows=10, cols=17, seed=seed)
            board, solution = gen.generate(target_coverage=target_coverage)

            # 실제로 풀어보기 (Greedy)
            env = FruitBoxEnv(FruitBoxConfig(rows=10, cols=17))
            env.board = board.astype(np.int16)

            obs = board.clip(0, 9).astype(np.int8)
            cleared = 0

            for _ in range(500):  # 최대 500 스텝
                legal = env.legal_actions()
                if len(legal) == 0:
                    break

                # Greedy: 가장 큰 것 선택
                sizes = [(env.rects[a][2]-env.rects[a][0]+1) *
                         (env.rects[a][3]-env.rects[a][1]+1) for a in legal]
                action = legal[np.argmax(sizes)]

                r1, c1, r2, c2 = env.rects[action]
                cells = (r2-r1+1) * (c2-c1+1)
                env.board[r1:r2+1, c1:c2+1] = 0
                cleared += cells

            actual_coverage = cleared / 170
            results.append(actual_coverage)

        avg_coverage = np.mean(results)
        std_coverage = np.std(results)
        print(f"  실제 달성: {avg_coverage:.1%} ± {std_coverage:.1%}")
        print(f"  목표 대비: {avg_coverage/target_coverage*100:.1f}%")
        print()

    print("=== 기존 랜덤 생성과 비교 ===")
    print()

    # 기존 방식
    print("기존 랜덤 생성:")
    old_results = []
    for seed in range(10):
        env = FruitBoxEnv(FruitBoxConfig(rows=10, cols=17))
        obs, info = env.reset(seed=seed)

        cleared = 0
        for _ in range(500):
            legal = env.legal_actions()
            if len(legal) == 0:
                break
            sizes = [(env.rects[a][2]-env.rects[a][0]+1) *
                     (env.rects[a][3]-env.rects[a][1]+1) for a in legal]
            action = legal[np.argmax(sizes)]
            r1, c1, r2, c2 = env.rects[action]
            cells = (r2-r1+1) * (c2-c1+1)
            env.board[r1:r2+1, c1:c2+1] = 0
            cleared += cells

        old_results.append(cleared / 170)

    print(f"  평균: {np.mean(old_results):.1%}")
    print()

    # 새 방식 (target_coverage=0.7)
    print("역방향 생성 (target=70%):")
    new_results = []
    for seed in range(10):
        gen = BackwardBoardGenerator(rows=10, cols=17, seed=seed)
        board, solution = gen.generate(target_coverage=0.7)

        env = FruitBoxEnv(FruitBoxConfig(rows=10, cols=17))
        env.board = board.astype(np.int16)

        cleared = 0
        for _ in range(500):
            legal = env.legal_actions()
            if len(legal) == 0:
                break
            sizes = [(env.rects[a][2]-env.rects[a][0]+1) *
                     (env.rects[a][3]-env.rects[a][1]+1) for a in legal]
            action = legal[np.argmax(sizes)]
            r1, c1, r2, c2 = env.rects[action]
            cells = (r2-r1+1) * (c2-c1+1)
            env.board[r1:r2+1, c1:c2+1] = 0
            cleared += cells

        new_results.append(cleared / 170)

    print(f"  평균: {np.mean(new_results):.1%}")
    print()
    print(f"개선: {np.mean(old_results):.1%} → {np.mean(new_results):.1%} "
          f"(+{(np.mean(new_results) - np.mean(old_results))*100:.1f}%p)")
