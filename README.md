# 🍎 AlphaApple - FruitBox Game AI

**한국 사과게임(FruitBox)을 해결하는 AI 에이전트**  
PPO 강화학습으로 평균 **77점** 달성 (베이스라인 대비 **+7.1% 개선**)

[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/kbsooo/AlphaApple)
[![Chrome Extension](https://img.shields.io/badge/🌐%20Chrome-Extension-blue)](#-chrome-확장-사용법)

## 🎯 완성된 기능들

✅ **AI 에이전트 학습 완료** (1M 스텝, 체크포인트별 성능 추적)  
✅ **ONNX 모델 변환** (2.95MB, 웹 배포 최적화)  
✅ **HuggingFace 모델 업로드** ([kbsooo/AlphaApple](https://huggingface.co/kbsooo/AlphaApple))  
✅ **Chrome 확장 프로그램** (실제 게임에서 실시간 AI 도움)  
✅ **성능 평가 도구** (Random/Greedy/PPO 비교 분석)

## 🚀 빠른 시작

### Chrome 확장으로 바로 사용하기

**1단계: 확장 설치** (30초)
1. Chrome에서 `chrome://extensions/` 접속
2. **개발자 모드** 켜기 (우상단 토글)
3. **압축해제된 확장 프로그램을 로드합니다** 클릭
4. `chrome_extension/` 폴더 선택
5. 🍎 **AlphaApple** 확장 설치 완료!

**2단계: 게임에서 사용**
1. 사과게임 웹사이트 접속
2. 우하단 🍎 **AI Assistant** 버튼 클릭  
3. AI가 추천하는 **금색 하이라이트** 영역을 드래그 선택
4. 점수 상승 확인! 🎉

**테스트:** `web/test_game.html` 파일로 확장 기능 테스트 가능

### 개발자용 모델 테스트

```bash
# 의존성 설치
uv install

# 전체 에이전트 성능 비교
python src/full_comparison.py

# AI 플레이 과정 관찰  
python src/evaluate_ppo.py --interactive

# ONNX 모델 변환 및 테스트
python src/convert_to_onnx.py
python src/test_onnx.py
```

## 📊 성능 결과

### 에이전트 비교 (100 에피소드)
| 에이전트 | 평균 점수 | 표준편차 | 개선율 |
|----------|-----------|----------|--------|
| Random   | 71.9      | 8.31     | -      |
| Greedy   | 73.3      | 8.48     | +1.9%  |
| **PPO**  | **77.0**  | **9.09** | **+7.1%** |

### 학습 과정
- **학습 스텝:** 1,000,000 (병렬 8환경)
- **체크포인트:** 10만 스텝마다 저장
- **최종 성과:** 76.6점 → 77점 (안정적 성능)

## 🌐 웹 배포 및 API

### HuggingFace 모델 사용

**PyTorch 모델:**
```python
from stable_baselines3 import PPO
model = PPO.load("pytorch_model.zip")
action, _ = model.predict(observation)
```

**ONNX 웹 배포:**
```javascript
import { InferenceSession } from 'onnxruntime-web';

const session = await InferenceSession.create(
    'https://huggingface.co/kbsooo/AlphaApple/resolve/main/fruitbox_ppo.onnx'
);

const result = await session.run({
    board_input: new ort.Tensor('float32', boardData, [1, 17, 10, 1])
});
```

### 직접 사용 (오프라인)
```bash
# ONNX 모델 다운로드
curl -L https://huggingface.co/kbsooo/AlphaApple/resolve/main/fruitbox_ppo.onnx -o model.onnx

# JavaScript 라이브러리 복사
cp web/fruitbox_ai.js ./your_project/
```

## 🎮 Chrome 확장 세부사항

### 지원 기능
- **자동 보드 감지**: 17×10 격자 자동 인식
- **실시간 AI 추천**: 최적 직사각형 하이라이트  
- **수동 보드 선택**: 자동 감지 실패 시 수동 지정
- **성능 표시**: AI 신뢰도 및 예상 점수

### 파일 구조
```
chrome_extension/
├── manifest.json          # 확장 설정
├── popup.html/js          # 제어 팝업  
├── content_script.js      # 게임 페이지 스크립트
├── fruitbox_ai.js         # AI 추론 라이브러리
├── models/fruitbox_ppo.onnx # AI 모델 (2.95MB)
└── libs/ort-web.min.js    # ONNX Runtime
```

## 📁 프로젝트 구조

```
alphaapple/
├── envs/fruitbox_env.py           # 게임 환경 구현
├── train/train_maskable_ppo.py    # PPO 학습 스크립트  
├── src/
│   ├── full_comparison.py         # 전체 성능 비교
│   ├── evaluate_ppo.py            # PPO 모델 평가
│   ├── convert_to_onnx.py         # ONNX 변환
│   └── upload_to_hf.py            # HuggingFace 업로드
├── chrome_extension/              # Chrome 확장 완성본
├── web/                           # 웹 배포 도구
└── ckpts/                         # 학습된 모델들
```

## 🔬 기술적 설계 세부사항

<details>
<summary>환경 및 행동 공간 설계</summary>

### 상태 (Observation)
- **형태:** 17×10 격자, 각 칸은 1-9 정수
- **텐서:** `[1, 17, 10]` int8 또는 원-핫 `[9, 17, 10]`
- **전처리:** 0-1 정규화 후 `(17, 10, 1)` 채널 차원 추가

### 행동 (Action) 
- **정의:** 모든 직사각형 (r1,c1,r2,c2) 조합 (r1≤r2, c1≤c2)
- **총 개수:** 8,415개 (17×18/2 × 10×11/2)
- **마스킹:** 합≠10 또는 빈칸 포함 시 선택 불가

### 보상 (Reward)
- **성공:** 제거된 셀 개수만큼 양수 보상
- **실패:** 보상 0 (상태 변화 없음)
- **추가:** 스텝 패널티 -0.01, 완료 보너스 +20

</details>

<details>
<summary>신경망 구조</summary>

### SmallGridCNN
```python
# 입력: (batch, 17, 10, 1)
Conv2d(1, 16, 3x3, padding=1) → ReLU
Conv2d(16, 32, 3x3, padding=1) → ReLU  
Conv2d(32, 32, 3x3, stride=2) → ReLU  # 다운샘플
Flatten → Linear(n_flatten, 128) → ReLU
# 출력: (batch, 128) 특징벡터
```

### PPO 정책
- **정책 헤드:** Linear(128 → 8415) + Softmax + Masking
- **가치 헤드:** Linear(128 → 1)
- **하이퍼파라미터:** lr=3e-4, γ=0.995, clip=0.2

</details>

## 🔧 개발 환경 설정

```bash
# 프로젝트 클론
git clone [repository-url]
cd alphaapple

# 의존성 설치 (uv 권장)
uv install

# 또는 pip 사용
pip install -e .
```

## 📝 TODO / 향후 계획

- [ ] **성능 개선**: Transformer 기반 모델 실험
- [ ] **커리큘럼 학습**: 6×7 → 10×17 점진적 확대  
- [ ] **모델 양자화**: INT8 최적화로 더 빠른 웹 추론
- [ ] **게임 사이트 연동**: 실제 사과게임 사이트 지원 확대
- [ ] **모바일 앱**: React Native로 모바일 버전

## 🤝 기여 방법

1. Fork 후 feature branch 생성
2. 변경사항 커밋
3. Pull Request 생성

## 📄 라이선스

MIT License

## 📞 문의

- **HuggingFace:** https://huggingface.co/kbsooo/AlphaApple
- **Issues:** GitHub Issues 활용
- **모델 사용 문의:** HuggingFace 모델 페이지 코멘트

---

**Made with 🍎 by kbsooo | Powered by PPO & ONNX**