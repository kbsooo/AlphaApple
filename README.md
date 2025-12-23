# 🍎 AlphaApple - RL for Perfect FruitBox Play

**목표**: 사과게임(FruitBox) 170개 셀 **전부 제거** (100% 클리어)

강화학습으로 인간을 넘어서는 성능 달성을 목표로 하는 프로젝트입니다.

## 🚀 현재 진행 상황 및 성과
- **DQN 베이스라인 구축 완료**: CNN 기반의 DQN 모델과 커리큘럼 학습을 통해 안정적인 학습 기반을 마련했습니다.
- **성능 기록**: 약 10,000 에피소드 학습 결과, 평균 **96% (163.4개)**의 사과를 제거하는 성과를 달성했습니다.
- **솔루션 보장형 환경**: `BackwardBoardGenerator`를 도입하여 항상 해답이 존재하는 보드에서 학습할 수 있도록 환경을 개선했습니다.

## 🧪 최근 변경 사항 (안정화/일반화 개선)
- **모델 안정화**: `src/models.py`에 BatchNorm을 추가하여 activation 폭발과 비현실적 Q-value 스케일 문제를 완화했습니다.
- **환경 일반화**: 보드 생성에 backward/random 혼합 비율(`backward_generator_ratio`)을 도입해 분포 이동을 줄였습니다.
- **보상 설계**: 미래 가능성(유효 액션 수 변화), 큰 영역 제거, 완전 클리어 보너스를 옵션으로 추가했습니다.
- **Colab + Drive 학습**: Google Drive에 체크포인트를 저장하는 스크립트 `experiments/train_colab_drive.py`를 추가했습니다.
- **환경 단일 소스**: 환경 코드는 `envs/fruitbox_env_improved.py`가 기준이며, `envs/fruitbox_env.py`는 호환용 래퍼입니다.

## 🛠 주요 기능
- **고성능 환경 (`envs/fruitbox_env_improved.py`)**: Prefix Sum 및 Incremental Action Masking을 적용하여 연산 속도를 극대화했습니다.
- **DQN 에이전트 (`src/agent.py`, `src/models.py`)**: 10채널 One-hot 인코딩 입력과 액션 마스킹을 지원하는 CNN 모델입니다.
- **Colab 최적화**: GPU 및 TPU 가속을 지원하는 학습 스크립트/노트북(`experiments/train_colab_drive.py`, `experiments/train_colab_integrated.ipynb`)을 제공합니다.
- **시각화 도구**: 에이전트의 플레이를 단계별 ASCII 그래픽으로 렌더링하고 전략을 분석할 수 있는 기능을 포함하고 있습니다.

## 📁 프로젝트 구조
- `envs/`: 사과게임 환경 및 보드 생성기
- `src/`: DQN 모델 아키텍처 및 에이전트 로직
- `experiments/`: 로컬 및 Colab용 학습 스크립트/노트북
- `checkpoints/`: 학습된 모델 저장 폴더

## 🚀 모델 배포 및 실전 도입
### 1. ONNX 변환 및 Hugging Face 업로드
- **ONNX 변환**: 브라우저에서 실행 가능하도록 모델을 변환합니다.
  ```bash
  uv run python src/export_onnx.py --model_path checkpoints/model.pth --output_path extension/model.onnx
  ```
- **Hugging Face 업로드**: 학습된 가중치와 ONNX 모델을 허브에 공유합니다.
  ```bash
  uv run python src/upload_hf.py --repo_id "사용자/리포지토리" --model_path checkpoints/model.pth --onnx_path extension/model.onnx
  ```

### 2. Chrome Extension (FruitBox Solver)
실제 [Gamesaien Fruit Box](https://en.gamesaien.com/game/fruit_box/) 사이트에서 모델을 실행하여 해답을 찾아주는 확장 프로그램입니다.

#### 설치 방법:
1. 브라우저 주소창에 `chrome://extensions/` 입력
2. '개발자 모드' 활성화
3. '압축해제된 확장 프로그램을 로드합니다' 클릭 후 프로젝트의 `extension/` 폴더 선택
4. **중요**: 확장 프로그램 폴더 안에 `model.onnx` 파일과 [onnxruntime-web](https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/onnxruntime.min.js) 라이브러리가 포함되어야 합니다.

#### 사용 방법:
- 게임 사이트 접속 후 확장 프로그램 팝업에서 **"Find Best Move"** 버튼 클릭
- 화면에 최적의 사과 박스가 빨간색으로 표시됩니다.

## 📈 앞으로의 계획
- **100% 클리어 도전**: 현재의 96% 성과를 넘어 100% 클리어를 위해 더 깊은 신경망(ResNet 등)과 PPO 알고리즘 도입을 검토 중입니다.
- **JAX/TPU 가속 확대**: 더 빠른 실험을 위해 JAX 기반의 분산 학습 환경을 고도화할 예정입니다.
