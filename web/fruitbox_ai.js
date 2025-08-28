// fruitbox_ai.js
/**
 * 웹에서 FruitBox AI 모델을 사용하는 예제
 * ONNX Runtime Web을 사용하여 브라우저에서 모델 추론
 */

class FruitBoxAI {
    constructor(modelUrl = './fruitbox_ppo.onnx') {
        this.modelUrl = modelUrl;
        this.session = null;
        this.isReady = false;
    }

    async initialize() {
        try {
            console.log('Loading FruitBox AI model...');
            
            // ONNX Runtime Web 세션 생성
            this.session = await ort.InferenceSession.create(this.modelUrl);
            this.isReady = true;
            
            console.log('FruitBox AI model loaded successfully!');
            console.log('Model inputs:', this.session.inputNames);
            console.log('Model outputs:', this.session.outputNames);
            
        } catch (error) {
            console.error('Failed to load AI model:', error);
            throw error;
        }
    }

    /**
     * 게임 보드에서 최적의 행동 예측
     * @param {number[][]} board - 17x10 격자 (1-9 숫자, 0은 빈 칸)
     * @returns {Promise<{action: number, confidence: number}>}
     */
    async predictAction(board) {
        if (!this.isReady) {
            throw new Error('Model not initialized. Call initialize() first.');
        }

        // 입력 전처리: 17x10 배열 → (1, 17, 10, 1) 텐서
        const boardArray = this.preprocessBoard(board);
        
        try {
            // ONNX 추론 실행
            const feeds = {
                [this.session.inputNames[0]]: new ort.Tensor('float32', boardArray, [1, 17, 10, 1])
            };
            
            const results = await this.session.run(feeds);
            const actionLogits = results[this.session.outputNames[0]].data;
            
            // 가장 높은 확률의 행동 선택
            const actionIndex = this.argMax(actionLogits);
            const confidence = this.softmax(actionLogits)[actionIndex];
            
            return {
                action: actionIndex,
                confidence: confidence,
                rectangle: this.actionToRectangle(actionIndex)
            };
            
        } catch (error) {
            console.error('Prediction failed:', error);
            throw error;
        }
    }

    /**
     * 보드 전처리: 정규화 및 형태 변환
     */
    preprocessBoard(board) {
        const data = new Float32Array(17 * 10 * 1);
        
        for (let r = 0; r < 17; r++) {
            for (let c = 0; c < 10; c++) {
                // 0-9 값을 0.0-1.0으로 정규화
                const value = board[r][c] / 9.0;
                data[r * 10 + c] = value;
            }
        }
        
        return data;
    }

    /**
     * 행동 인덱스를 직사각형 좌표로 변환
     * 학습 시 사용된 매핑 규칙과 동일해야 함
     */
    actionToRectangle(actionIndex) {
        // 원본 환경과 동일한 매핑 로직
        const rectangles = [];
        for (let r1 = 0; r1 < 17; r1++) {
            for (let r2 = r1; r2 < 17; r2++) {
                for (let c1 = 0; c1 < 10; c1++) {
                    for (let c2 = c1; c2 < 10; c2++) {
                        rectangles.push({r1, c1, r2, c2});
                    }
                }
            }
        }
        
        return rectangles[actionIndex];
    }

    /**
     * 직사각형이 유효한지 확인 (합이 10인지)
     */
    isValidRectangle(board, rect) {
        let sum = 0;
        for (let r = rect.r1; r <= rect.r2; r++) {
            for (let c = rect.c1; c <= rect.c2; c++) {
                if (board[r][c] === 0) return false;  // 빈 칸 포함 시 무효
                sum += board[r][c];
            }
        }
        return sum === 10;
    }

    /**
     * 유효한 행동들만 필터링해서 예측
     */
    async predictValidAction(board) {
        const prediction = await this.predictAction(board);
        
        // 예측된 행동이 유효하지 않으면 다른 방법 시도
        if (!this.isValidRectangle(board, prediction.rectangle)) {
            console.warn('Predicted action is invalid, finding alternative...');
            
            // 모든 유효한 행동을 찾아서 그 중 최고 점수 선택
            const validActions = this.findValidActions(board);
            if (validActions.length > 0) {
                // 가장 큰 직사각형 선택 (greedy fallback)
                const bestAction = validActions.reduce((best, current) => {
                    const bestSize = (best.r2 - best.r1 + 1) * (best.c2 - best.c1 + 1);
                    const currentSize = (current.r2 - current.r1 + 1) * (current.c2 - current.c1 + 1);
                    return currentSize > bestSize ? current : best;
                });
                
                return {
                    action: -1,  // 대체 행동임을 표시
                    rectangle: bestAction,
                    confidence: 0.5
                };
            }
        }
        
        return prediction;
    }

    /**
     * 보드에서 모든 유효한 행동 찾기
     */
    findValidActions(board) {
        const valid = [];
        
        for (let r1 = 0; r1 < 17; r1++) {
            for (let r2 = r1; r2 < 17; r2++) {
                for (let c1 = 0; c1 < 10; c1++) {
                    for (let c2 = c1; c2 < 10; c2++) {
                        const rect = {r1, c1, r2, c2};
                        if (this.isValidRectangle(board, rect)) {
                            valid.push(rect);
                        }
                    }
                }
            }
        }
        
        return valid;
    }

    // 유틸리티 함수들
    argMax(array) {
        return array.indexOf(Math.max(...array));
    }

    softmax(logits) {
        const maxLogit = Math.max(...logits);
        const exp = logits.map(x => Math.exp(x - maxLogit));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sum);
    }
}

// 사용 예제
async function example() {
    // 모델 초기화
    const ai = new FruitBoxAI('./fruitbox_ppo.onnx');
    await ai.initialize();

    // 예제 보드 (17x10)
    const board = [
        [2, 1, 9, 7, 9, 6, 4, 4, 1, 4],
        [1, 8, 5, 1, 7, 7, 5, 2, 1, 1],
        // ... 15 more rows
    ];

    // AI 예측
    const prediction = await ai.predictValidAction(board);
    console.log('AI recommendation:', prediction);
    
    // 직사각형 좌표로 게임 실행
    const rect = prediction.rectangle;
    console.log(`Select rectangle: (${rect.r1},${rect.c1}) to (${rect.r2},${rect.c2})`);
}

// 전역으로 클래스 노출 (브라우저에서 사용)
if (typeof window !== 'undefined') {
    window.FruitBoxAI = FruitBoxAI;
}

// Node.js에서 사용
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FruitBoxAI;
}