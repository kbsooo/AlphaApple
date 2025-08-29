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
        this.gameDetected = false;
        this.highlightedElements = [];
    }

    async initialize() {
        try {
            console.log('🍎 AlphaApple: Initializing FruitBox AI...');
            
            // UI 먼저 생성 (AI 로딩과 무관하게)
            this.createOverlay();
            this.addControlButton();
            
            // AI 모델 로드 시도 (실패해도 계속 진행)
            try {
                this.ai = new FruitBoxAI(chrome.runtime.getURL('models/fruitbox_ppo.onnx'));
                await this.ai.initialize();
                console.log('✅ AI model loaded successfully');
            } catch (aiError) {
                console.warn('⚠️ AI model failed to load, running in board-detection-only mode:', aiError.message);
                this.ai = null;
            }
            
            // 게임 보드 감지 (초기 시도)
            this.detectGameBoard();
            
            // Play 버튼 클릭 감지를 위한 리스너 추가
            this.setupGameStartListener();
            
            console.log('🍎 AlphaApple: Ready! (AI:', this.ai ? 'Enabled' : 'Disabled', ')');
            
        } catch (error) {
            console.error('🍎 AlphaApple: Critical initialization failed:', error);
            // 최소한의 UI라도 표시
            this.createOverlay();
            this.showError('Initialization failed: ' + error.message);
        }
    }

    /**
     * Play 버튼 클릭 감지 및 게임 시작 후 재감지
     */
    setupGameStartListener() {
        console.log('🍎 Setting up game start detection...');
        
        // Play 버튼들 감지
        const checkForPlayButton = () => {
            const playSelectors = [
                'button[onclick*="play"]', 
                'button[onclick*="start"]', 
                '.play-button', 
                '#play-btn',
                'input[value*="Play"]',
                'a[href*="play"]'
            ];
            
            for (const selector of playSelectors) {
                const playButton = document.querySelector(selector);
                if (playButton) {
                    console.log('🍎 Found play button:', selector);
                    playButton.addEventListener('click', () => {
                        console.log('🍎 Play button clicked! Re-detecting board in 3 seconds...');
                        setTimeout(() => this.detectGameBoard(), 3000);
                        setTimeout(() => this.detectGameBoard(), 6000); // 두 번째 시도
                    });
                }
            }
        };
        
        checkForPlayButton();
        
        // DOM 변경 감지 (새로운 요소 추가 시)
        const observer = new MutationObserver((mutations) => {
            let shouldRedetect = false;
            
            mutations.forEach(mutation => {
                if (mutation.addedNodes.length > 0) {
                    mutation.addedNodes.forEach(node => {
                        if (node.nodeType === 1) { // Element node
                            // 새로운 캔버스나 테이블이 추가되면
                            if (node.tagName === 'CANVAS' || node.tagName === 'TABLE') {
                                shouldRedetect = true;
                            }
                            // 또는 많은 요소가 한번에 추가되면 (게임 로딩)
                            if (node.children && node.children.length > 10) {
                                shouldRedetect = true;
                            }
                        }
                    });
                }
            });
            
            if (shouldRedetect && !this.gameDetected) {
                console.log('🍎 DOM changes detected, re-detecting board...');
                setTimeout(() => this.detectGameBoard(), 1000);
            }
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    /**
     * 게임 보드 감지 - CreateJS 우선, DOM 백업
     */
    detectGameBoard() {
        console.log('🍎 Starting enhanced board detection...');
        
        // 1단계: CreateJS 게임 데이터 직접 추출 시도
        const createJSBoard = this.extractFromCreateJS();
        if (createJSBoard) {
            console.log('🍎 Found game board via CreateJS!');
            this.gameBoard = createJSBoard;
            this.gameDetected = true;
            this.showBoardPreview(createJSBoard, 'CreateJS');
            return;
        }
        
        // 2단계: 일반적인 DOM 선택자들 시도
        const commonSelectors = [
            '.game-board', '.fruit-grid', '.puzzle-board', '#gameBoard',
            'table.game', '.grid', '#game-container', '.board-container',
            'canvas[width][height]', '.game-canvas', '#canvas'
        ];

        for (const selector of commonSelectors) {
            const element = document.querySelector(selector);
            if (element && this.attemptBoardParse(element)) {
                console.log(`🍎 Found game board: ${selector}`);
                return;
            }
        }

        // 3단계: 더 광범위한 검색
        console.log('🍎 Trying comprehensive detection...');
        this.tryComprehensiveDetection();
    }

    /**
     * CreateJS에서 게임 보드 직접 추출
     */
    extractFromCreateJS() {
        console.log('🍎 === CreateJS 게임 구조 분석 중... ===');
        const results = {
            potentialBoards: [],
            analysis: {}
        };

        // 1. CreateJS Stage 분석
        if (window.stage) {
            console.log('✅ window.stage 발견!');
            console.log('Stage type:', stage.constructor.name);
            console.log('Stage children count:', stage.children?.length || 0);
            
            results.analysis.stage = {
                type: stage.constructor.name,
                childrenCount: stage.children?.length || 0
            };

            // Stage children 상세 분석
            if (stage.children) {
                stage.children.forEach((child, i) => {
                    console.log(`  Stage Child ${i}: ${child.constructor.name}`);
                    const boardData = this.searchObjectForBoard(child, `stage.children[${i}]`, 0, 3);
                    if (boardData) {
                        results.potentialBoards.push(boardData);
                    }
                });
            }
        }

        // 2. ExportRoot 분석
        if (window.exportRoot) {
            console.log('✅ window.exportRoot 발견!');
            console.log('ExportRoot type:', exportRoot.constructor.name);
            
            results.analysis.exportRoot = { type: exportRoot.constructor.name };
            
            // ExportRoot 객체 탐색
            const boardData = this.searchObjectForBoard(exportRoot, 'exportRoot', 0, 4);
            if (boardData) {
                results.potentialBoards.push(boardData);
            }
        }

        // 3. 게임 관련 전역 변수 찾기
        const gameKeywords = ['board', 'grid', 'map', 'level', 'puzzle', 'fruit', 'data', 'game'];
        
        for (const prop in window) {
            const lowerProp = prop.toLowerCase();
            for (const keyword of gameKeywords) {
                if (lowerProp.includes(keyword)) {
                    try {
                        const value = window[prop];
                        if (value && typeof value === 'object' && Array.isArray(value)) {
                            if (this.couldBeGameBoard(value)) {
                                console.log(`🎯 게임 보드 후보 발견! window.${prop}`);
                                results.potentialBoards.push({
                                    path: `window.${prop}`,
                                    data: value,
                                    confidence: this.rateBoardQuality(value),
                                    size: `${value.length}x${value[0]?.length || 0}`
                                });
                            }
                        }
                    } catch (e) {
                        // 접근 불가
                    }
                    break;
                }
            }
        }

        // 가장 좋은 후보 선택
        if (results.potentialBoards.length > 0) {
            results.potentialBoards.sort((a, b) => b.confidence - a.confidence);
            const best = results.potentialBoards[0];
            
            console.log(`🍎 최고 후보: ${best.path} (confidence: ${best.confidence}, size: ${best.size})`);
            console.log('보드 데이터:', best.data);
            
            return best.data;
        }

        console.log('❌ CreateJS에서 게임 보드를 찾지 못했습니다');
        return null;
    }

    /**
     * 객체를 재귀적으로 탐색해서 보드 데이터 찾기
     */
    searchObjectForBoard(obj, path, depth = 0, maxDepth = 3) {
        if (!obj || depth > maxDepth) return null;

        try {
            // 직접 배열인지 확인
            if (Array.isArray(obj)) {
                if (this.couldBeGameBoard(obj)) {
                    console.log(`🎯 보드 후보 발견: ${path}`);
                    return {
                        path,
                        data: obj,
                        confidence: this.rateBoardQuality(obj),
                        size: `${obj.length}x${obj[0]?.length || 0}`
                    };
                }
            }

            // 객체의 프로퍼티들 탐색
            for (const key in obj) {
                if (key.length > 20) continue; // 너무 긴 키는 스킵
                
                try {
                    const value = obj[key];
                    
                    // 배열인지 확인
                    if (Array.isArray(value)) {
                        if (this.couldBeGameBoard(value)) {
                            console.log(`🎯 보드 후보 발견: ${path}.${key}`);
                            return {
                                path: `${path}.${key}`,
                                data: value,
                                confidence: this.rateBoardQuality(value),
                                size: `${value.length}x${value[0]?.length || 0}`
                            };
                        }
                    }
                    // 객체면 재귀 탐색
                    else if (value && typeof value === 'object' && depth < maxDepth) {
                        const result = this.searchObjectForBoard(value, `${path}.${key}`, depth + 1, maxDepth);
                        if (result) return result;
                    }
                } catch (e) {
                    // 접근 불가능한 프로퍼티 스킵
                }
            }
        } catch (e) {
            console.warn(`🍎 Error searching ${path}:`, e);
        }

        return null;
    }

    /**
     * 배열이 게임 보드일 가능성 체크
     */
    couldBeGameBoard(arr) {
        if (!Array.isArray(arr) || arr.length === 0) return false;
        
        // 2D 배열이고 적당한 크기인지 체크
        const firstRow = arr[0];
        if (!Array.isArray(firstRow)) return false;
        
        const rows = arr.length;
        const cols = firstRow.length;
        
        // 게임 보드 같은 크기인지 (8x8 ~ 25x20 정도)
        if (rows < 8 || rows > 25 || cols < 8 || cols > 20) return false;
        
        // 모든 행이 같은 길이인지 체크
        for (let i = 1; i < Math.min(5, rows); i++) {
            if (!Array.isArray(arr[i]) || arr[i].length !== cols) return false;
        }
        
        // 숫자 데이터인지 체크
        let numberCells = 0;
        let totalCells = 0;
        
        for (let r = 0; r < Math.min(3, rows); r++) {
            const row = arr[r];
            for (let c = 0; c < Math.min(5, cols); c++) {
                totalCells++;
                const cell = row[c];
                if (typeof cell === 'number' && cell >= 0 && cell <= 9) {
                    numberCells++;
                }
            }
        }
        
        const ratio = numberCells / totalCells;
        return ratio >= 0.6; // 60% 이상이 유효한 숫자
    }

    /**
     * 보드 품질 평가 (더 좋은 후보 선택용)
     */
    rateBoardQuality(board) {
        let score = 0;
        
        // 크기 점수 - 17x10이 이상적
        if (board.length === 17) score += 20;
        else if (board.length >= 15 && board.length <= 20) score += 10;
        else if (board.length >= 10 && board.length <= 25) score += 5;
        
        if (board[0] && board[0].length === 10) score += 20;
        else if (board[0] && board[0].length >= 8 && board[0].length <= 12) score += 10;
        else if (board[0] && board[0].length >= 6 && board[0].length <= 15) score += 5;

        // 숫자 분포 점수
        const allNumbers = [];
        board.forEach(row => {
            if (Array.isArray(row)) {
                row.forEach(cell => {
                    if (typeof cell === 'number' && cell >= 0 && cell <= 9) {
                        allNumbers.push(cell);
                    }
                });
            }
        });

        const uniqueNumbers = new Set(allNumbers).size;
        if (uniqueNumbers >= 7) score += 15; // 1-9 범위의 다양한 숫자
        else if (uniqueNumbers >= 5) score += 10;
        else if (uniqueNumbers >= 3) score += 5;

        // 데이터 밀도 점수 (0이 아닌 셀의 비율)
        const nonZeroCells = allNumbers.filter(n => n > 0).length;
        const totalCells = board.length * (board[0]?.length || 0);
        const density = nonZeroCells / totalCells;
        
        if (density >= 0.3 && density <= 0.8) score += 10; // 적당한 밀도
        else if (density >= 0.1 && density <= 0.9) score += 5;

        return score;
    }

    /**
     * 보드 미리보기 표시
     */
    showBoardPreview(board, source) {
        let existing = document.getElementById('board-preview-display');
        if (existing) existing.remove();

        const display = document.createElement('div');
        display.id = 'board-preview-display';
        display.style.cssText = `
            position: fixed;
            top: 60px;
            left: 20px;
            background: rgba(0, 100, 0, 0.95);
            color: white;
            padding: 12px;
            border-radius: 8px;
            z-index: 999998;
            font-family: monospace;
            font-size: 10px;
            line-height: 1.2;
            max-width: 280px;
            border: 2px solid #4CAF50;
        `;

        let content = `🍎 게임 보드 감지 성공! (${source})\n크기: ${board.length}x${board[0]?.length || 0}\n\n`;
        
        // 처음 몇 행만 표시
        for (let r = 0; r < Math.min(6, board.length); r++) {
            const row = board[r];
            if (Array.isArray(row)) {
                content += row.slice(0, 10).map(cell => 
                    String(cell).substring(0, 1)
                ).join(' ') + '\n';
            }
        }
        
        if (board.length > 6) {
            content += `... (+${board.length - 6}행 더)`;
        }

        display.innerHTML = `
            <pre style="margin: 0; color: #90EE90;">${content}</pre>
            <div style="margin-top: 8px; font-size: 9px;">
                <button onclick="this.parentElement.parentElement.remove()" style="background: #2E7D32; color: white; border: none; padding: 3px 6px; border-radius: 3px;">닫기</button>
                <button onclick="console.log('🍎 Current Board:', window.globalHelper.gameBoard)" style="background: #1976D2; color: white; border: none; padding: 3px 6px; border-radius: 3px; margin-left: 3px;">콘솔로그</button>
            </div>
        `;

        document.body.appendChild(display);
        
        // 10초 후 자동 제거
        setTimeout(() => {
            if (display.parentNode) display.remove();
        }, 10000);
    }

    /**
     * 더 광범위한 보드 감지 시도
     */
    tryComprehensiveDetection() {
        // 모든 테이블 검사
        const tables = document.querySelectorAll('table');
        console.log(`🍎 Found ${tables.length} tables`);
        
        for (const table of tables) {
            if (this.attemptBoardParse(table)) {
                console.log('🍎 Board detected in table!');
                return;
            }
        }

        // 그리드 패턴 div 검사
        const gridDivs = document.querySelectorAll('div[class*="grid"], div[class*="board"], div[class*="game"]');
        console.log(`🍎 Found ${gridDivs.length} grid divs`);
        
        for (const div of gridDivs) {
            if (this.attemptBoardParse(div)) {
                console.log('🍎 Board detected in div!');
                return;
            }
        }

        // Canvas 검사 (게임이 Canvas로 구현된 경우)
        const canvases = document.querySelectorAll('canvas');
        console.log(`🍎 Found ${canvases.length} canvas elements`);
        
        for (const canvas of canvases) {
            if (canvas.width >= 300 && canvas.height >= 400) {
                console.log('🍎 Large canvas detected - might be game board');
                this.handleCanvasGame(canvas);
                return;
            }
        }

        // 숫자가 많이 포함된 요소 검사
        this.detectByNumbers();
        
        if (!this.gameDetected) {
            console.log('🍎 Game board not found. Showing manual selector...');
            this.showManualSelector();
        }
    }

    /**
     * 숫자 패턴으로 보드 감지 시도
     */
    detectByNumbers() {
        // 1-9 숫자가 많이 포함된 요소들 찾기
        const allElements = document.querySelectorAll('*');
        const candidates = [];
        
        for (const el of allElements) {
            const text = el.textContent || '';
            const numbers = text.match(/[1-9]/g);
            
            if (numbers && numbers.length >= 50) {  // 최소 50개 숫자
                const children = el.children.length;
                candidates.push({ element: el, numberCount: numbers.length, childCount: children });
            }
        }
        
        // 가장 유력한 후보 선택
        candidates.sort((a, b) => b.numberCount - a.numberCount);
        
        console.log(`🍎 Found ${candidates.length} number-rich elements`);
        
        for (const candidate of candidates.slice(0, 3)) {
            console.log(`🍎 Trying candidate with ${candidate.numberCount} numbers, ${candidate.childCount} children`);
            if (this.attemptBoardParse(candidate.element)) {
                console.log('🍎 Board detected by number pattern!');
                return;
            }
        }
    }

    /**
     * Canvas 게임 처리
     */
    handleCanvasGame(canvas) {
        console.log('🍎 Canvas game detected - manual mode required');
        this.gameDetected = true;
        
        // Canvas는 수동 입력 필요함을 알림
        this.overlay.innerHTML = `
            <div>🍎 <strong>AlphaApple</strong></div>
            <div>🖼️ Canvas game detected</div>
            <div>📝 Manual board input required</div>
            <button id="manual-input">✏️ Input Board</button>
        `;
        this.overlay.style.display = 'block';
        
        document.getElementById('manual-input')?.addEventListener('click', () => {
            this.showManualBoardInput();
        });
    }

    /**
     * 보드 파싱 시도 (성공하면 true 반환)
     */
    attemptBoardParse(element) {
        try {
            // 다양한 셀 선택자 시도
            const cellSelectors = [
                'td', '.cell', '.grid-item', '.game-cell',
                'div[class*="cell"]', 'span[class*="cell"]',
                '.square', '.tile', '.box'
            ];
            
            for (const selector of cellSelectors) {
                const cells = element.querySelectorAll(selector);
                console.log(`🍎 Trying selector '${selector}': ${cells.length} cells`);
                
                // 170개 (17x10) 또는 비슷한 수의 셀들
                if (cells.length >= 150 && cells.length <= 200) {
                    const boardData = this.extractBoardData(cells, 17, 10);
                    
                    // 보드 데이터 유효성 검사
                    if (this.validateBoardData(boardData)) {
                        this.gameBoard = boardData;
                        this.gameDetected = true;
                        this.boardElement = element;
                        this.cells = cells;
                        
                        // 감지된 보드에 시각적 표시
                        element.style.outline = '3px solid #4CAF50';
                        
                        return true;
                    }
                }
            }
            
            return false;
        } catch (error) {
            console.error('🍎 Board parse attempt failed:', error);
            return false;
        }
    }

    /**
     * 보드 데이터 유효성 검사
     */
    validateBoardData(board) {
        if (!board || board.length !== 17) return false;
        
        let numberCount = 0;
        let validNumbers = 0;
        
        for (let r = 0; r < 17; r++) {
            if (!board[r] || board[r].length !== 10) return false;
            
            for (let c = 0; c < 10; c++) {
                numberCount++;
                if (board[r][c] >= 1 && board[r][c] <= 9) {
                    validNumbers++;
                }
            }
        }
        
        const validRatio = validNumbers / numberCount;
        console.log(`🍎 Board validation: ${validNumbers}/${numberCount} valid (${(validRatio*100).toFixed(1)}%)`);
        
        return validRatio >= 0.5;  // 최소 50% 유효한 숫자
    }

    /**
     * DOM에서 게임 보드 데이터 추출
     */
    parseGameBoard(element) {
        try {
            const cells = element.querySelectorAll('td, .cell, .grid-item');
            
            if (cells.length === 170) {  // 17x10 = 170
                this.gameBoard = this.extractBoardData(cells, 17, 10);
                this.gameDetected = true;
                console.log('🍎 Board parsed successfully!', this.gameBoard);
            } else {
                console.log(`🍎 Unexpected cell count: ${cells.length} (expected 170)`);
                this.gameDetected = false;
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
        button.className = 'fruitbox-ai-button';
        document.body.appendChild(button);
        
        this.controlButton = button;
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
     * AI 추천 표시 (null 안전)
     */
    async showRecommendation() {
        // Overlay null 체크
        if (!this.overlay) {
            console.error('🍎 Overlay not initialized, creating it now');
            this.createOverlay();
            if (!this.overlay) {
                console.error('🍎 Failed to create overlay');
                return;
            }
        }

        // 게임 보드나 AI가 없는 경우
        if (!this.gameBoard) {
            this.overlay.innerHTML = `
                <div>🍎 <strong>AlphaApple</strong></div>
                <div>❌ Game board not detected</div>
                <div><small>Try clicking Play button</small></div>
                <button onclick="window.globalHelper.detectGameBoard()">🔄 Detect Board</button>
            `;
            this.overlay.style.display = 'block';
            return;
        }

        if (!this.ai) {
            this.overlay.innerHTML = `
                <div>🍎 <strong>AlphaApple</strong></div>
                <div>✅ Board detected: ${this.gameBoard.length}x${this.gameBoard[0]?.length || 0}</div>
                <div>⚠️ AI disabled (WASM load failed)</div>
                <div><small>Board detection working!</small></div>
                <button onclick="window.globalHelper.detectGameBoard()">🔄 Refresh</button>
            `;
            this.overlay.style.display = 'block';
            return;
        }

        // AI 분석 시작
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
            const refreshBtn = document.getElementById('refresh-ai');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => {
                    this.detectGameBoard();
                    this.showRecommendation();
                });
            }
            
        } catch (error) {
            this.overlay.innerHTML = `
                <div>🍎 <strong>AlphaApple</strong></div>
                <div>❌ AI prediction failed</div>
                <div><small>${error.message}</small></div>
                <button onclick="window.globalHelper.detectGameBoard()">🔄 Try Again</button>
            `;
        }
    }

    /**
     * 에러 표시 (새 함수)
     */
    showError(message) {
        if (!this.overlay) {
            this.createOverlay();
        }
        
        if (this.overlay) {
            this.overlay.innerHTML = `
                <div>🍎 <strong>AlphaApple</strong></div>
                <div>❌ Error</div>
                <div><small>${message}</small></div>
            `;
            this.overlay.style.display = 'block';
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
            color: black;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 10002;
            font-family: Arial, sans-serif;
        `;
        
        selector.innerHTML = `
            <h3>🍎 AlphaApple Setup</h3>
            <p><strong>게임 보드를 찾을 수 없습니다.</strong></p>
            <div style="margin: 15px 0;">
                <button id="debug-info" style="margin: 5px; padding: 8px;">🔍 디버깅 정보</button>
                <button id="manual-select" style="margin: 5px; padding: 8px;">👆 보드 직접 선택</button>
                <button id="manual-input" style="margin: 5px; padding: 8px;">✏️ 수동 입력</button>
            </div>
            <button id="cancel-setup" style="background: #f44336; color: white; border: none; padding: 8px 15px; border-radius: 5px;">취소</button>
        `;
        
        document.body.appendChild(selector);
        
        // 디버깅 정보 버튼
        document.getElementById('debug-info').addEventListener('click', () => {
            this.showDebugInfo();
        });
        
        // 수동 선택 버튼
        document.getElementById('manual-select').addEventListener('click', () => {
            selector.remove();
            this.startManualSelection();
        });
        
        // 수동 입력 버튼
        document.getElementById('manual-input').addEventListener('click', () => {
            selector.remove();
            this.showManualBoardInput();
        });
        
        // 취소 버튼
        document.getElementById('cancel-setup').addEventListener('click', () => {
            selector.remove();
        });
    }

    /**
     * 디버깅 정보 표시
     */
    showDebugInfo() {
        const tables = document.querySelectorAll('table').length;
        const canvases = document.querySelectorAll('canvas').length;
        const divs = document.querySelectorAll('div').length;
        
        // 숫자가 많은 요소들 찾기
        const numberElements = [];
        document.querySelectorAll('*').forEach(el => {
            const text = el.textContent || '';
            const numbers = text.match(/[1-9]/g);
            if (numbers && numbers.length >= 20) {
                numberElements.push({
                    tag: el.tagName,
                    class: el.className,
                    id: el.id,
                    numbers: numbers.length,
                    text: text.substring(0, 50) + '...'
                });
            }
        });
        
        const debugInfo = `
            <h4>🔍 페이지 분석 결과</h4>
            <p><strong>발견된 요소들:</strong></p>
            <ul>
                <li>테이블: ${tables}개</li>
                <li>캔버스: ${canvases}개</li>
                <li>div 요소: ${divs}개</li>
                <li>숫자가 많은 요소: ${numberElements.length}개</li>
            </ul>
            
            ${numberElements.length > 0 ? `
                <p><strong>숫자가 많은 요소들:</strong></p>
                ${numberElements.slice(0, 3).map(el => `
                    <div style="font-size: 12px; margin: 5px 0; background: #f5f5f5; padding: 5px;">
                        <strong>${el.tag}</strong> 
                        ${el.class ? `class="${el.class}"` : ''} 
                        ${el.id ? `id="${el.id}"` : ''}
                        <br>숫자 ${el.numbers}개: ${el.text}
                    </div>
                `).join('')}
            ` : ''}
            
            <p style="font-size: 12px; color: #666;">
                개발자 도구(F12)를 열어서 게임 보드의 HTML 구조를 확인하고,
                해당 요소를 직접 클릭해보세요.
            </p>
        `;
        
        const existingDialog = document.querySelector('.debug-dialog');
        if (existingDialog) existingDialog.remove();
        
        const dialog = document.createElement('div');
        dialog.className = 'debug-dialog';
        dialog.style.cssText = `
            position: fixed; top: 20px; left: 20px; 
            background: white; color: black;
            padding: 20px; border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 10003; max-width: 400px;
            font-family: Arial, sans-serif; font-size: 14px;
        `;
        
        dialog.innerHTML = debugInfo + '<button onclick="this.parentElement.remove()">닫기</button>';
        document.body.appendChild(dialog);
    }

    /**
     * 수동 선택 모드 시작
     */
    startManualSelection() {
        document.body.style.cursor = 'crosshair';
        
        const instruction = document.createElement('div');
        instruction.style.cssText = `
            position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.8); color: white;
            padding: 15px; border-radius: 8px;
            z-index: 10004; text-align: center;
        `;
        instruction.innerHTML = `
            <div>🎯 <strong>게임 보드를 클릭하세요</strong></div>
            <div style="font-size: 12px; margin-top: 5px;">
                게임이 표시된 영역(테이블, 그리드 등)을 클릭
            </div>
        `;
        document.body.appendChild(instruction);
        
        // 클릭 이벤트 리스너
        const clickHandler = (e) => {
            const clicked = e.target;
            console.log('🍎 Clicked element:', clicked.tagName, clicked.className);
            
            // 클릭된 요소와 그 부모들을 시도
            const candidates = [clicked];
            let parent = clicked.parentElement;
            
            for (let i = 0; i < 5 && parent; i++) {
                candidates.push(parent);
                parent = parent.parentElement;
            }
            
            for (const candidate of candidates) {
                if (this.attemptBoardParse(candidate)) {
                    console.log('🍎 Manual selection successful!');
                    break;
                }
            }
            
            document.body.style.cursor = '';
            instruction.remove();
            e.preventDefault();
            e.stopPropagation();
        };
        
        document.addEventListener('click', clickHandler, {once: true});
        
        // 5초 후 자동 취소
        setTimeout(() => {
            document.body.style.cursor = '';
            instruction.remove();
        }, 5000);
    }

    /**
     * 수동 보드 입력 다이얼로그
     */
    showManualBoardInput() {
        const dialog = document.createElement('div');
        dialog.style.cssText = `
            position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background: white; color: black;
            padding: 20px; border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 10005; font-family: Arial, sans-serif;
            max-width: 500px;
        `;
        
        dialog.innerHTML = `
            <h3>🍎 보드 수동 입력</h3>
            <p>현재 게임 보드의 숫자들을 입력하세요 (17행 × 10열):</p>
            <textarea id="board-input" rows="17" cols="25" style="font-family: monospace; width: 100%; margin: 10px 0;"
                placeholder="예시:
2 1 9 7 9 6 4 4 1 4
1 8 5 1 7 7 5 2 1 1
3 5 4 9 3 7 1 7 6 7
..."></textarea>
            <div>
                <button id="parse-manual">✅ 파싱하기</button>
                <button id="cancel-manual">❌ 취소</button>
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 10px;">
                각 행은 10개 숫자를 공백으로 구분, 총 17행<br>
                빈 칸은 0으로 입력
            </div>
        `;
        
        document.body.appendChild(dialog);
        
        document.getElementById('parse-manual').addEventListener('click', () => {
            const input = document.getElementById('board-input').value;
            this.parseManualInput(input);
            dialog.remove();
        });
        
        document.getElementById('cancel-manual').addEventListener('click', () => {
            dialog.remove();
        });
    }

    /**
     * 수동 입력 파싱
     */
    parseManualInput(input) {
        try {
            const lines = input.trim().split('\n');
            if (lines.length !== 17) {
                alert('❌ 17행이 필요합니다');
                return;
            }
            
            const board = [];
            for (const line of lines) {
                const numbers = line.trim().split(/\s+/).map(n => parseInt(n) || 0);
                if (numbers.length !== 10) {
                    alert('❌ 각 행마다 10개 숫자가 필요합니다');
                    return;
                }
                board.push(numbers);
            }
            
            this.gameBoard = board;
            this.gameDetected = true;
            
            console.log('🍎 Manual board input successful!', board);
            alert('✅ 보드 입력 완료! 이제 AI 추천을 받을 수 있습니다.');
            
        } catch (error) {
            console.error('🍎 Manual input failed:', error);
            alert('❌ 입력 형식이 올바르지 않습니다');
        }
    }

    /**
     * 하이라이트 제거
     */
    clearHighlights() {
        this.highlightedElements.forEach(el => {
            el.classList.remove('ai-highlight');
        });
        this.highlightedElements = [];
    }

    /**
     * 직사각형 영역 하이라이트
     */
    highlightRectangle(rect) {
        this.clearHighlights();
        
        // 실제 DOM 셀들에 하이라이트 적용 (간단한 예시)
        try {
            const cells = document.querySelectorAll('td, .cell, .grid-item');
            for (let r = rect.r1; r <= rect.r2; r++) {
                for (let c = rect.c1; c <= rect.c2; c++) {
                    const cellIndex = r * 10 + c;
                    if (cellIndex < cells.length) {
                        const cell = cells[cellIndex];
                        cell.classList.add('ai-highlight');
                        this.highlightedElements.push(cell);
                    }
                }
            }
        } catch (error) {
            console.error('🍎 Highlight failed:', error);
        }
    }
}

// 전역 helper 인스턴스
let globalHelper = null;

// 팝업과의 메시지 통신
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (!globalHelper) {
        sendResponse({gameDetected: false, aiEnabled: false, message: "Helper not initialized"});
        return;
    }

    switch(request.action) {
        case 'getStatus':
            sendResponse({
                gameDetected: globalHelper.gameDetected,
                aiEnabled: globalHelper.isEnabled,
                message: globalHelper.gameDetected ? 
                    (globalHelper.isEnabled ? "AI Active" : "AI Ready") : 
                    "Game not detected"
            });
            break;
            
        case 'enableAI':
            globalHelper.isEnabled = true;
            globalHelper.showRecommendation();
            sendResponse({
                gameDetected: globalHelper.gameDetected,
                aiEnabled: true,
                message: "AI Enabled"
            });
            break;
            
        case 'disableAI':
            globalHelper.isEnabled = false;
            globalHelper.overlay.style.display = 'none';
            globalHelper.clearHighlights();
            sendResponse({
                gameDetected: globalHelper.gameDetected,
                aiEnabled: false,
                message: "AI Disabled"
            });
            break;
            
        case 'refreshBoard':
            globalHelper.detectGameBoard();
            sendResponse({
                gameDetected: globalHelper.gameDetected,
                aiEnabled: globalHelper.isEnabled,
                message: globalHelper.gameDetected ? "Board refreshed" : "No game found"
            });
            break;
    }
});

// 페이지 로드 시 자동 초기화
(async () => {
    // 페이지가 완전히 로드될 때까지 대기
    if (document.readyState === 'loading') {
        await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve));
    }
    
    // 추가로 1초 대기 (동적 콘텐츠 로딩 고려)
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // FruitBox 헬퍼 초기화
    globalHelper = new FruitBoxHelper();
    await globalHelper.initialize();
})();