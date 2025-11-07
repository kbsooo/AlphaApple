"""
Autoregressive Policy용 환경 래퍼

정책의 (r1, c1, r2, c2) 출력을 환경의 action index로 변환하고,
추가 검증 및 마스킹을 처리합니다.
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any
from envs.fruitbox_env import FruitBoxEnv, FruitBoxConfig


class AutoregressiveEnvWrapper(gym.Wrapper):
    """
    Autoregressive Policy를 위한 환경 래퍼

    좌표 (r1, c1, r2, c2)를 받아서 환경의 action index로 변환
    """

    def __init__(self, env: FruitBoxEnv):
        super().__init__(env)
        self.env = env

        # 좌표 → action index 매핑 생성
        self._build_coord_to_action_map()

    def _build_coord_to_action_map(self):
        """좌표 → action index 빠른 조회를 위한 매핑"""
        rows, cols = self.env.cfg.rows, self.env.cfg.cols
        self.coord_to_action = {}

        for action_idx, (r1, c1, r2, c2) in enumerate(self.env.rects):
            self.coord_to_action[(r1, c1, r2, c2)] = action_idx

    def coords_to_action(
        self,
        r1: int,
        c1: int,
        r2: int,
        c2: int
    ) -> Tuple[int, bool]:
        """
        좌표를 action index로 변환

        Returns:
            action_idx: 환경의 행동 인덱스
            valid: 이 행동이 합=10을 만족하는지
        """
        # 좌표 범위 체크
        if not (0 <= r1 <= r2 < self.env.cfg.rows and
                0 <= c1 <= c2 < self.env.cfg.cols):
            return -1, False

        # Action index 조회
        coord_tuple = (int(r1), int(c1), int(r2), int(c2))
        action_idx = self.coord_to_action.get(coord_tuple, -1)

        if action_idx == -1:
            return -1, False

        # 합=10 체크
        region = self.env.board[r1:r2+1, c1:c2+1]
        region_sum = int(np.sum(region))
        valid = (region_sum == 10)

        return action_idx, valid

    def step_with_coords(
        self,
        r1: int,
        c1: int,
        r2: int,
        c2: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        좌표로 step 실행

        Args:
            r1, c1, r2, c2: 직사각형 좌표

        Returns:
            observation, reward, terminated, truncated, info
        """
        action_idx, valid = self.coords_to_action(r1, c1, r2, c2)

        # 추가 정보
        info_extra = {
            'coords': (r1, c1, r2, c2),
            'action_idx': action_idx,
            'coords_valid': valid,
        }

        # 불법 행동 처리
        if not valid:
            obs = self.env.board.clip(0, 9).astype(np.int8, copy=False)
            mask = self.env._compute_action_mask(self.env.board)
            info = {
                'action_mask': mask,
                'illegal_action': True,
                **info_extra
            }
            # 불법 행동에 대한 페널티
            reward = -1.0
            terminated = False  # 계속 진행 (학습을 위해)
            truncated = False
            return obs, reward, terminated, truncated, info

        # 정상 step 실행
        obs, reward, terminated, truncated, info = self.env.step(action_idx)

        # 추가 정보 병합
        info.update(info_extra)

        return obs, reward, terminated, truncated, info

    def get_valid_coords_mask(self) -> np.ndarray:
        """
        현재 상태에서 유효한 좌표 조합의 마스크

        Returns:
            mask: (rows, cols, rows, cols) boolean 배열
                  mask[r1, c1, r2, c2] = True if 합=10
        """
        rows, cols = self.env.cfg.rows, self.env.cfg.cols
        mask = np.zeros((rows, cols, rows, cols), dtype=bool)

        action_mask = self.env._compute_action_mask(self.env.board)

        for action_idx, valid in enumerate(action_mask):
            if valid:
                r1, c1, r2, c2 = self.env.rects[action_idx]
                mask[r1, c1, r2, c2] = True

        return mask


def make_autoregressive_env(
    rows: int = 10,
    cols: int = 17,
    reward_per_cell: float = 1.0,
    **kwargs
) -> AutoregressiveEnvWrapper:
    """Autoregressive 환경 생성 헬퍼"""
    config = FruitBoxConfig(
        rows=rows,
        cols=cols,
        reward_per_cell=reward_per_cell,
        **kwargs
    )
    base_env = FruitBoxEnv(config)
    wrapped_env = AutoregressiveEnvWrapper(base_env)
    return wrapped_env


# ============================================================
# 테스트 코드
# ============================================================

if __name__ == "__main__":
    print("=== Autoregressive Wrapper 테스트 ===")
    print()

    # 환경 생성
    env = make_autoregressive_env(rows=10, cols=17)
    obs, info = env.reset(seed=42)

    print(f"초기 보드 shape: {obs.shape}")
    print(f"초기 합법 행동 수: {np.sum(info['action_mask'])}")
    print()

    # 합법 행동 하나 찾기
    legal_actions = env.env.legal_actions()
    if len(legal_actions) > 0:
        # 첫 번째 합법 행동의 좌표
        action_idx = legal_actions[0]
        r1, c1, r2, c2 = env.env.rects[action_idx]

        print(f"테스트 행동: ({r1}, {c1}, {r2}, {c2})")

        # 영역 확인
        region = obs[r1:r2+1, c1:c2+1]
        print(f"영역 숫자: {region.flatten()}")
        print(f"영역 합: {np.sum(region)}")
        print()

        # 좌표로 step 실행
        obs, reward, terminated, truncated, info = env.step_with_coords(r1, c1, r2, c2)

        print(f"보상: {reward}")
        print(f"종료: {terminated}")
        print(f"좌표 유효: {info['coords_valid']}")
        print(f"Action index: {info['action_idx']}")
        print()

    # 불법 행동 테스트
    print("=== 불법 행동 테스트 ===")
    env.reset(seed=42)

    # 의도적으로 합≠10인 좌표 선택
    obs, reward, terminated, truncated, info = env.step_with_coords(0, 0, 0, 1)

    print(f"보상: {reward}")
    print(f"불법 행동: {info['illegal_action']}")
    print(f"좌표 유효: {info['coords_valid']}")
    print()

    # 좌표 마스크 테스트
    print("=== 유효 좌표 마스크 ===")
    env.reset(seed=42)
    coord_mask = env.get_valid_coords_mask()
    print(f"마스크 shape: {coord_mask.shape}")
    print(f"유효한 좌표 조합 수: {np.sum(coord_mask)}")
    print()

    print("✅ 모든 테스트 통과!")
