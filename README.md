# 🍎 AlphaApple - RL for Perfect FruitBox Play

**목표**: 사과게임(FruitBox) 170개 셀 **전부 제거** (100% 클리어)

강화학습으로 인간을 넘어서는 성능 달성을 목표로 하는 프로젝트입니다.

## 🚀 현재 진행 상황 및 성과
- **DQN 베이스라인 구축 완료**: CNN 기반의 DQN 모델과 커리큘럼 학습을 통해 안정적인 학습 기반을 마련했습니다.
- **성능 기록**: 약 10,000 에피소드 학습 결과, 평균 **96% (163.4개)**의 사과를 제거하는 성과를 달성했습니다.
- **솔루션 보장형 환경**: `BackwardBoardGenerator`를 도입하여 항상 해답이 존재하는 보드에서 학습할 수 있도록 환경을 개선했습니다.

## 🛠 주요 기능
- **고성능 환경 (`envs/fruitbox_env.py`)**: Prefix Sum 및 Incremental Action Masking을 적용하여 연산 속도를 극대화했습니다.
- **DQN 에이전트 (`src/agent.py`, `src/models.py`)**: 10채널 One-hot 인코딩 입력과 액션 마스킹을 지원하는 CNN 모델입니다.
- **Colab 최적화**: GPU 및 TPU 가속을 지원하는 통합 학습 노트북(`experiments/train_colab_integrated.ipynb`, `experiments/train_colab_jax.ipynb`)을 제공합니다.
- **시각화 도구**: 에이전트의 플레이를 단계별 ASCII 그래픽으로 렌더링하고 전략을 분석할 수 있는 기능을 포함하고 있습니다.

## 📁 프로젝트 구조
- `envs/`: 사과게임 환경 및 보드 생성기
- `src/`: DQN 모델 아키텍처 및 에이전트 로직
- `experiments/`: 로컬 및 Colab용 학습 스크립트/노트북
- `checkpoints/`: 학습된 모델 저장 폴더

## 📈 앞으로의 계획
- **100% 클리어 도전**: 현재의 96% 성과를 넘어 100% 클리어를 위해 더 깊은 신경망(ResNet 등)과 PPO 알고리즘 도입을 검토 중입니다.
- **JAX/TPU 가속 확대**: 더 빠른 실험을 위해 JAX 기반의 분산 학습 환경을 고도화할 예정입니다.