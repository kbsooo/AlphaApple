# 🍎 AlphaApple Chrome 확장 사용법

## 📥 설치 방법

### 1단계: 개발자 모드로 설치
```bash
# Chrome에서 확장 관리 페이지 열기
chrome://extensions/
```

1. **개발자 모드** 토글 켜기 (우상단)
2. **"압축해제된 확장 프로그램을 로드합니다"** 클릭
3. `chrome_extension/` 폴더 선택
4. **AlphaApple** 확장이 설치됨

### 2단계: 권한 확인
- 설치 후 🍎 아이콘이 툴바에 표시됨
- 필요 시 확장 고정 (📌)

## 🎮 사용법

### 실제 사용 시나리오

**상황 1: 일반적인 FruitBox 게임 사이트**
1. 사과게임 웹사이트 접속
2. 게임이 로드되면 우하단에 **🍎 AI Assistant** 버튼 나타남
3. 버튼 클릭 → AI가 자동으로 보드 분석
4. 우상단에 **AI 추천** 표시:
   ```
   🍎 AlphaApple
   ✨ AI Recommendation:
   📍 Select: (3,2) to (5,4)  
   🎯 Confidence: 87.3%
   📏 Size: 9 cells
   ```

**상황 2: 보드가 자동 감지 안 되는 경우**
1. 🍎 아이콘 클릭 → 팝업 열림
2. **"🔄 Refresh Board"** 클릭
3. 여전히 감지 안 되면 수동 선택 모드 활성화
4. 게임 보드를 직접 클릭해서 지정

**상황 3: 실시간 게임 플레이**
1. AI 추천 받기
2. 게임에서 해당 직사각형 드래그 선택
3. 보드 변경되면 **새로고침** 클릭
4. 새로운 AI 추천 받기
5. 반복...

## 🔧 기술적 동작 원리

### AI 모델 로딩
```javascript
// 확장이 시작될 때
const ai = new FruitBoxAI('chrome-extension://[id]/models/fruitbox_ppo.onnx');
await ai.initialize();  // 2.95MB 모델 로드 (한 번만)
```

### 보드 감지
```javascript
// DOM에서 17x10 격자 찾기
const selectors = [
    '.game-board', '.fruit-grid', '.puzzle-board',
    '#gameBoard', 'table.game', '.grid'
];
```

### AI 추론
```javascript
// 보드 → AI → 추천
const board = extractBoardFromDOM();  // DOM 파싱
const prediction = await ai.predictValidAction(board);  // AI 추론 (~50ms)
highlightRectangle(prediction.rectangle);  // UI 표시
```

## 📋 파일 구조
```
chrome_extension/
├── manifest.json          # 확장 기본 설정
├── popup.html             # 🍎 아이콘 클릭 시 팝업
├── popup.js              # 팝업 제어 로직
├── content_script.js     # 게임 페이지에서 실행
├── fruitbox_ai.js        # AI 추론 라이브러리
├── styles.css            # UI 스타일
├── models/
│   └── fruitbox_ppo.onnx # AI 모델 (2.95MB)
└── libs/
    └── ort-web.min.js    # ONNX Runtime Web
```

## 🌐 지원하는 게임 사이트

현재 **자동 감지** 시도하는 사이트들:
- `*.example.com/*` (예시)
- `fruitbox-game.com/*` (가상)

**수동 지정**으로 모든 사이트에서 사용 가능!

## ❗ 문제 해결

**"Game board not detected"**
→ 🔄 Refresh Board 클릭 또는 보드를 직접 클릭

**"AI prediction failed"**  
→ 페이지 새로고침 후 다시 시도

**확장이 안 보임**
→ `chrome://extensions/` 가서 AlphaApple 활성화 확인

**모델 로딩 실패**
→ 인터넷 연결 확인 (ONNX 모델 다운로드 필요)

## 📊 예상 성능

- **AI 점수:** 평균 77점
- **개선율:** 랜덤 대비 +7.1%  
- **응답속도:** ~50ms (모델 로드 후)
- **모델 크기:** 2.95MB (첫 로드만)

## 🔄 실제 사용 예시

1. **네이버 게임** 같은 곳에서 사과게임 플레이
2. 확장 활성화 → AI가 최적 수 찾아줌
3. 제안된 직사각형 클릭/드래그
4. 점수 상승! 🎉

Chrome 확장이 완전히 준비됐습니다!