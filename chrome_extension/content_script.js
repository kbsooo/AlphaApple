// content_script.js
/**
 * FruitBox 게임 페이지에서 DOM을 분석하고 AI 추천을 제공하는 Content Script
 */

class FruitBoxHelper {
    constructor() {
        this.ai = null;
        this.gameBoard = null;
        this.isEnabled = false;
        this.overlay = null;
    }

    async initialize() {
        try {
            console.log('🍎 AlphaApple: Initializing FruitBox AI...');
            
            // AI 모델 로드
            this.ai = new FruitBoxAI(chrome.runtime.getURL('models/fruitbox_ppo.onnx'));
            await this.ai.initialize();
            
            // 게임 보드 감지
            this.detectGameBoard();
            
            // UI 추가
            this.createOverlay();
            this.addControlButton();
            
            console.log('🍎 AlphaApple: Ready!');
            
        } catch (error) {
            console.error('🍎 AlphaApple: Initialization failed:', error);
        }
    }

    /**
     * 게임 보드 DOM 요소 감지 및 파싱
     */
    detectGameBoard() {
        // 일반적인 게임 보드 선택자들 시도
        const selectors = [
            '.game-board',
            '.fruit-grid', 
            '.puzzle-board',
            '#gameBoard',
            'table.game',
            '.grid'
        ];

        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) {
                console.log(`🍎 Found game board: ${selector}`);
                this.parseGameBoard(element);
                return;
            }
        }

        console.log('🍎 Game board not found. Trying manual detection...');
        this.tryManualDetection();
    }

    /**
     * DOM에서 게임 보드 데이터 추출
     */
    parseGameBoard(element) {
        try {
            const cells = element.querySelectorAll('td, .cell, .grid-item');
            
            if (cells.length === 170) {  // 17x10 = 170
                this.gameBoard = this.extractBoardData(cells, 17, 10);
                console.log('🍎 Board parsed successfully!', this.gameBoard);
            } else {
                console.log(`🍎 Unexpected cell count: ${cells.length} (expected 170)`);
            }
        } catch (error) {
            console.error('🍎 Board parsing failed:', error);
        }
    }

    /**
     * DOM 셀들에서 숫자 데이터 추출
     */
    extractBoardData(cells, rows, cols) {
        const board = Array(rows).fill().map(() => Array(cols).fill(0));
        
        for (let i = 0; i < cells.length; i++) {
            const cell = cells[i];
            const row = Math.floor(i / cols);
            const col = i % cols;
            
            // 셀에서 숫자 추출 (여러 방법 시도)
            let value = 0;
            
            const text = cell.textContent || cell.innerText;
            const numMatch = text.match(/\d+/);
            if (numMatch) {
                value = parseInt(numMatch[0]);
            }
            
            // data 속성도 확인
            if (value === 0 && cell.dataset.value) {
                value = parseInt(cell.dataset.value);
            }
            
            // 빈 칸이나 이미 제거된 칸은 0
            if (cell.classList.contains('empty') || cell.classList.contains('cleared')) {
                value = 0;
            }
            
            board[row][col] = value;
        }
        
        return board;
    }

    /**
     * AI 추천 표시 오버레이 생성
     */
    createOverlay() {
        this.overlay = document.createElement('div');
        this.overlay.id = 'fruitbox-ai-overlay';
        this.overlay.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            z-index: 10000;
            max-width: 250px;
            display: none;
        `;
        
        document.body.appendChild(this.overlay);
    }

    /**
     * AI 활성화/비활성화 버튼 추가
     */
    addControlButton() {
        const button = document.createElement('button');
        button.textContent = '🍎 AI Assistant';
        button.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            z-index: 10001;
            font-weight: bold;
        `;
        
        button.addEventListener('click', () => this.toggleAI());
        document.body.appendChild(button);
    }

    /**
     * AI 도움 켜기/끄기
     */
    async toggleAI() {
        this.isEnabled = !this.isEnabled;
        
        if (this.isEnabled) {
            await this.showRecommendation();
        } else {
            this.overlay.style.display = 'none';
        }
    }

    /**
     * AI 추천 표시
     */
    async showRecommendation() {
        if (!this.gameBoard || !this.ai) {
            this.overlay.innerHTML = `
                <div>🍎 <strong>AlphaApple</strong></div>
                <div>❌ Game board not detected</div>
                <div><small>Try refreshing the page</small></div>
            `;
            this.overlay.style.display = 'block';
            return;
        }

        this.overlay.innerHTML = `
            <div>🍎 <strong>AlphaApple</strong></div>
            <div>🤔 Analyzing board...</div>
        `;
        this.overlay.style.display = 'block';

        try {
            const prediction = await this.ai.predictValidAction(this.gameBoard);
            const rect = prediction.rectangle;
            
            this.overlay.innerHTML = `
                <div>🍎 <strong>AlphaApple</strong></div>
                <div>✨ <strong>AI Recommendation:</strong></div>
                <div>📍 Select: (${rect.r1+1},${rect.c1+1}) to (${rect.r2+1},${rect.c2+1})</div>
                <div>🎯 Confidence: ${(prediction.confidence * 100).toFixed(1)}%</div>
                <div>📏 Size: ${(rect.r2-rect.r1+1) * (rect.c2-rect.c1+1)} cells</div>
                <button id="refresh-ai">🔄 Refresh</button>
            `;
            
            // 새로고침 버튼
            document.getElementById('refresh-ai').addEventListener('click', () => {
                this.detectGameBoard();
                this.showRecommendation();
            });
            
        } catch (error) {
            this.overlay.innerHTML = `
                <div>🍎 <strong>AlphaApple</strong></div>
                <div>❌ AI prediction failed</div>
                <div><small>${error.message}</small></div>
            `;
        }
    }

    /**
     * 수동 보드 감지 시도
     */
    tryManualDetection() {
        // 페이지의 모든 테이블이나 그리드 구조 찾기
        const tables = document.querySelectorAll('table');
        const grids = document.querySelectorAll('div[class*="grid"], div[class*="board"]');
        
        console.log(`🍎 Found ${tables.length} tables, ${grids.length} grid divs`);
        
        // 사용자가 수동으로 보드를 지정할 수 있도록 UI 제공
        this.showManualSelector();
    }

    showManualSelector() {
        const selector = document.createElement('div');
        selector.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 10002;
        `;
        
        selector.innerHTML = `
            <h3>🍎 AlphaApple Setup</h3>
            <p>Click on the game board to enable AI assistant:</p>
            <button id="cancel-setup">Cancel</button>
        `;
        
        document.body.appendChild(selector);
        
        // 클릭으로 보드 선택
        document.addEventListener('click', (e) => {
            if (e.target.closest('#cancel-setup')) {
                selector.remove();
                return;
            }
            
            // 클릭된 요소가 게임 보드인지 확인
            const clicked = e.target.closest('table, .grid, .board');
            if (clicked) {
                this.parseGameBoard(clicked);
                selector.remove();
                e.preventDefault();
            }
        }, {once: true});
    }
}

// 페이지 로드 시 자동 초기화
(async () => {
    // 페이지가 완전히 로드될 때까지 대기
    if (document.readyState === 'loading') {
        await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve));
    }
    
    // 추가로 1초 대기 (동적 콘텐츠 로딩 고려)
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // FruitBox 헬퍼 초기화
    const helper = new FruitBoxHelper();
    await helper.initialize();
})();