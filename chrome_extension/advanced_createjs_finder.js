// 🍎 고급 CreateJS 변수 탐지 스크립트
// console에서 실행하거나 content script에 통합

console.log('🍎 === 고급 CreateJS 변수 탐색 시작 ===');

function findCreateJSVariables() {
    const results = {
        found: {},
        methods: [],
        gameBoards: []
    };

    // 1. 직접 전역 변수 체크
    console.log('\n🔍 1단계: 직접 전역 변수 체크');
    const directVars = ['stage', 'exportRoot', 'canvas', 'anim_container'];
    directVars.forEach(varName => {
        if (window[varName]) {
            console.log(`✅ window.${varName} 발견:`, window[varName]);
            results.found[varName] = window[varName];
            results.methods.push(`window.${varName}`);
        } else {
            console.log(`❌ window.${varName} 없음`);
        }
    });

    // 2. CreateJS 라이브러리 인스턴스 찾기
    console.log('\n🔍 2단계: CreateJS 라이브러리 인스턴스');
    if (typeof createjs !== 'undefined') {
        console.log('✅ createjs 라이브러리 발견');
        
        // Stage 인스턴스들 찾기
        try {
            if (createjs.Stage && createjs.Stage._instances) {
                console.log('Stage instances:', createjs.Stage._instances);
                results.found.stageInstances = createjs.Stage._instances;
                results.methods.push('createjs.Stage._instances');
            }
        } catch (e) {
            console.log('Stage._instances 접근 실패:', e);
        }

        // 활성 ticker 확인
        try {
            if (createjs.Ticker) {
                console.log('Ticker 정보:', createjs.Ticker);
                results.found.ticker = createjs.Ticker;
            }
        } catch (e) {
            console.log('Ticker 접근 실패:', e);
        }
    }

    // 3. Canvas element에서 연결된 객체 찾기
    console.log('\n🔍 3단계: Canvas element 연결 객체');
    const canvas = document.getElementById('canvas');
    if (canvas) {
        console.log('✅ Canvas element 발견:', canvas);
        
        // Canvas의 모든 프로퍼티 중 stage 관련 찾기
        Object.getOwnPropertyNames(canvas).forEach(prop => {
            if (prop.toLowerCase().includes('stage')) {
                console.log(`Canvas.${prop}:`, canvas[prop]);
                results.found[`canvas_${prop}`] = canvas[prop];
                results.methods.push(`canvas.${prop}`);
            }
        });

        // Canvas context에서 찾기
        const ctx = canvas.getContext('2d');
        if (ctx) {
            Object.getOwnPropertyNames(ctx).forEach(prop => {
                if (prop.toLowerCase().includes('stage')) {
                    console.log(`Context.${prop}:`, ctx[prop]);
                }
            });
        }
    }

    // 4. 함수 스코프에서 변수 찾기 (eval 사용)
    console.log('\n🔍 4단계: 함수 스코프 변수 탐색');
    const possibleVars = ['stage', 'exportRoot', 'lib', 'ss', 'stage_1'];
    possibleVars.forEach(varName => {
        try {
            const result = eval(varName);
            if (result) {
                console.log(`✅ ${varName} 발견 (eval):`, result);
                results.found[`eval_${varName}`] = result;
                results.methods.push(`eval('${varName}')`);
            }
        } catch (e) {
            // 변수가 없으면 에러 발생
        }
    });

    // 5. iframe이나 다른 context 체크
    console.log('\n🔍 5단계: iframe/다른 컨텍스트');
    const iframes = document.querySelectorAll('iframe');
    if (iframes.length > 0) {
        console.log(`iframe ${iframes.length}개 발견`);
        iframes.forEach((iframe, i) => {
            try {
                const iframeWindow = iframe.contentWindow;
                if (iframeWindow && iframeWindow.stage) {
                    console.log(`✅ iframe[${i}].stage 발견:`, iframeWindow.stage);
                    results.found[`iframe${i}_stage`] = iframeWindow.stage;
                }
            } catch (e) {
                console.log(`iframe[${i}] 접근 실패:`, e.message);
            }
        });
    }

    // 6. 게임 보드 데이터 직접 탐색
    console.log('\n🔍 6단계: 게임 보드 데이터 직접 탐색');
    
    // stage가 있으면 children 탐색
    Object.values(results.found).forEach((obj, i) => {
        if (obj && obj.children) {
            console.log(`Found object with children[${i}]:`, obj);
            searchForGameBoard(obj, `found_object_${i}`, results);
        }
    });

    return results;
}

// 게임 보드 탐색 헬퍼 함수
function searchForGameBoard(obj, path, results) {
    try {
        if (!obj || typeof obj !== 'object') return;

        // children 배열이 있으면 탐색
        if (Array.isArray(obj.children)) {
            obj.children.forEach((child, i) => {
                if (child && typeof child === 'object') {
                    // 자주 쓰이는 게임 보드 프로퍼티 이름들
                    const boardProps = ['board', 'grid', 'map', 'data', 'cells', 'tiles'];
                    boardProps.forEach(prop => {
                        if (child[prop] && Array.isArray(child[prop])) {
                            console.log(`🎯 ${path}.children[${i}].${prop} 게임보드 후보:`, child[prop]);
                            
                            if (looksLikeGameBoard(child[prop])) {
                                results.gameBoards.push({
                                    path: `${path}.children[${i}].${prop}`,
                                    data: child[prop],
                                    obj: child
                                });
                            }
                        }
                    });

                    // 재귀적으로 더 깊이 탐색 (제한적으로)
                    if (i < 10) { // 처음 10개만
                        searchForGameBoard(child, `${path}.children[${i}]`, results);
                    }
                }
            });
        }

        // 직접 프로퍼티도 확인
        const boardProps = ['board', 'grid', 'map', 'data', 'cells', 'tiles', 'level'];
        boardProps.forEach(prop => {
            if (obj[prop] && Array.isArray(obj[prop])) {
                if (looksLikeGameBoard(obj[prop])) {
                    console.log(`🎯 ${path}.${prop} 게임보드 발견:`, obj[prop]);
                    results.gameBoards.push({
                        path: `${path}.${prop}`,
                        data: obj[prop],
                        obj: obj
                    });
                }
            }
        });

    } catch (e) {
        console.warn(`게임보드 탐색 중 오류 (${path}):`, e);
    }
}

// 게임 보드인지 판단하는 함수
function looksLikeGameBoard(arr) {
    if (!Array.isArray(arr) || arr.length === 0) return false;
    
    // 2D 배열 체크
    const firstRow = arr[0];
    if (!Array.isArray(firstRow)) return false;
    
    const rows = arr.length;
    const cols = firstRow.length;
    
    // 적당한 크기 (5x5 ~ 25x20)
    if (rows < 5 || rows > 25 || cols < 5 || cols > 20) return false;
    
    // 숫자 데이터 비율 체크
    let numberCount = 0;
    let totalCount = 0;
    
    for (let r = 0; r < Math.min(3, rows); r++) {
        if (Array.isArray(arr[r])) {
            for (let c = 0; c < Math.min(5, cols); c++) {
                totalCount++;
                const cell = arr[r][c];
                if (typeof cell === 'number' && cell >= 0 && cell <= 9) {
                    numberCount++;
                }
            }
        }
    }
    
    return numberCount / totalCount >= 0.6; // 60% 이상 숫자
}

// 결과 분석 및 권장사항
function analyzeResults(results) {
    console.log('\n🍎 === 분석 결과 ===');
    console.log('발견된 객체:', Object.keys(results.found).length);
    console.log('접근 방법:', results.methods);
    console.log('게임 보드 후보:', results.gameBoards.length);

    if (results.gameBoards.length > 0) {
        console.log('\n🎯 가장 유력한 게임 보드:');
        results.gameBoards.forEach((board, i) => {
            console.log(`${i + 1}. ${board.path}`);
            console.log(`   크기: ${board.data.length}x${board.data[0]?.length || 0}`);
            console.log('   첫 3행:', board.data.slice(0, 3));
        });

        // 가장 좋은 후보 추천
        const best = results.gameBoards[0];
        console.log('\n🏆 권장 사용법:');
        console.log(`let gameBoard = ${best.path};`);
        console.log('console.log(gameBoard);');
    }

    return results;
}

// 실행
const searchResults = findCreateJSVariables();
const analysis = analyzeResults(searchResults);

// 전역에 저장
window.createJSSearchResults = analysis;

console.log('\n📝 사용법:');
console.log('window.createJSSearchResults로 결과 확인 가능');
console.log('게임보드 후보가 있으면 path 복사해서 직접 접근하세요!');