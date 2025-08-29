// 🍎 gamesaien.com 게임 구조 직접 분석 스크립트
// 개발자 콘솔에서 실행하여 CreateJS 객체들을 분석

console.log('🍎 게임 구조 분석 시작...');

function analyzeGameStructure() {
    const results = {
        stage: null,
        exportRoot: null,
        globalVars: [],
        canvasInfo: null,
        potentialBoards: []
    };

    // 1. CreateJS Stage 분석
    console.log('\n=== 1. CreateJS Stage 분석 ===');
    if (window.stage) {
        console.log('✅ window.stage 발견!');
        console.log('Stage type:', stage.constructor.name);
        console.log('Stage children count:', stage.children?.length || 0);
        
        results.stage = {
            type: stage.constructor.name,
            childrenCount: stage.children?.length || 0,
            children: stage.children ? stage.children.map((child, i) => ({
                index: i,
                type: child.constructor.name,
                hasChildren: !!child.children,
                childrenCount: child.children?.length || 0
            })) : []
        };

        // 각 stage children 상세 분석
        if (stage.children) {
            stage.children.forEach((child, i) => {
                console.log(`  Child ${i}: ${child.constructor.name}`);
                if (child.children) {
                    console.log(`    하위 children: ${child.children.length}개`);
                    child.children.forEach((subchild, j) => {
                        if (j < 5) { // 처음 5개만 표시
                            console.log(`      [${j}] ${subchild.constructor.name}`);
                        }
                    });
                    if (child.children.length > 5) {
                        console.log(`      ... +${child.children.length - 5}개 더`);
                    }
                }
            });
        }
    } else {
        console.log('❌ window.stage 없음');
    }

    // 2. ExportRoot 분석  
    console.log('\n=== 2. ExportRoot 분석 ===');
    if (window.exportRoot) {
        console.log('✅ window.exportRoot 발견!');
        console.log('ExportRoot type:', exportRoot.constructor.name);
        console.log('ExportRoot properties:');
        
        const props = Object.getOwnPropertyNames(exportRoot);
        props.forEach(prop => {
            try {
                const value = exportRoot[prop];
                if (value && typeof value === 'object') {
                    if (Array.isArray(value)) {
                        console.log(`  ${prop}: Array[${value.length}]`);
                        if (value.length > 0 && Array.isArray(value[0])) {
                            console.log(`    → 2D Array [${value.length}][${value[0].length}]`);
                            // 게임 보드 후보인지 검사
                            if (couldBeGameBoard(value)) {
                                console.log(`    🎯 게임 보드 후보 발견! exportRoot.${prop}`);
                                results.potentialBoards.push({
                                    path: `exportRoot.${prop}`,
                                    data: value,
                                    size: `${value.length}x${value[0]?.length || 0}`
                                });
                            }
                        }
                    } else if (value.constructor) {
                        console.log(`  ${prop}: ${value.constructor.name}`);
                    }
                }
            } catch (e) {
                console.log(`  ${prop}: (접근 불가)`);
            }
        });
        
        results.exportRoot = { type: exportRoot.constructor.name, hasData: true };
    } else {
        console.log('❌ window.exportRoot 없음');
    }

    // 3. 게임 관련 전역 변수 찾기
    console.log('\n=== 3. 게임 관련 전역 변수 탐색 ===');
    const gameKeywords = ['board', 'grid', 'map', 'level', 'puzzle', 'fruit', 'data', 'game', 'stage'];
    
    for (const prop in window) {
        const lowerProp = prop.toLowerCase();
        for (const keyword of gameKeywords) {
            if (lowerProp.includes(keyword)) {
                try {
                    const value = window[prop];
                    if (value && typeof value === 'object') {
                        if (Array.isArray(value)) {
                            console.log(`  window.${prop}: Array[${value.length}]`);
                            if (couldBeGameBoard(value)) {
                                console.log(`    🎯 게임 보드 후보 발견! window.${prop}`);
                                results.potentialBoards.push({
                                    path: `window.${prop}`,
                                    data: value,
                                    size: `${value.length}x${value[0]?.length || 0}`
                                });
                            }
                        } else {
                            console.log(`  window.${prop}: ${value.constructor?.name || typeof value}`);
                        }
                        results.globalVars.push(prop);
                    }
                } catch (e) {
                    // 접근 불가
                }
                break;
            }
        }
    }

    // 4. Canvas 정보 분석
    console.log('\n=== 4. Canvas 분석 ===');
    const canvases = document.querySelectorAll('canvas');
    console.log(`Canvas 개수: ${canvases.length}`);
    
    if (canvases.length > 0) {
        canvases.forEach((canvas, i) => {
            console.log(`  Canvas ${i}: ${canvas.width}x${canvas.height}`);
        });
        results.canvasInfo = Array.from(canvases).map(c => ({
            width: c.width,
            height: c.height
        }));
    }

    // 5. 게임 보드 후보들 상세 분석
    if (results.potentialBoards.length > 0) {
        console.log('\n=== 5. 게임 보드 후보 상세 분석 ===');
        results.potentialBoards.forEach((candidate, i) => {
            console.log(`\n후보 ${i + 1}: ${candidate.path} (${candidate.size})`);
            
            // 첫 몇 행 출력해보기
            const board = candidate.data;
            for (let r = 0; r < Math.min(3, board.length); r++) {
                if (Array.isArray(board[r])) {
                    const row = board[r].slice(0, 10).map(cell => {
                        if (typeof cell === 'number') return cell.toString();
                        if (typeof cell === 'string') return cell.substring(0, 1);
                        return '?';
                    });
                    console.log(`  행 ${r}: [${row.join(', ')}]`);
                }
            }
            
            // 데이터 타입 분석
            const flatData = board.flat();
            const types = {};
            flatData.slice(0, 50).forEach(cell => {
                const type = typeof cell;
                types[type] = (types[type] || 0) + 1;
            });
            console.log(`  데이터 타입:`, types);
        });
    }

    console.log('\n🍎 분석 완료! 결과:');
    console.log('결과 객체:', results);
    
    return results;
}

// 게임 보드일 가능성 체크
function couldBeGameBoard(arr) {
    if (!Array.isArray(arr) || arr.length === 0) return false;
    
    // 2D 배열이고 적당한 크기인지 체크
    const firstRow = arr[0];
    if (!Array.isArray(firstRow)) return false;
    
    const rows = arr.length;
    const cols = firstRow.length;
    
    // 게임 보드 같은 크기인지 (10x10 ~ 20x15 정도)
    if (rows < 8 || rows > 25 || cols < 8 || cols > 20) return false;
    
    // 모든 행이 같은 길이인지 체크
    for (let i = 1; i < Math.min(5, rows); i++) {
        if (!Array.isArray(arr[i]) || arr[i].length !== cols) return false;
    }
    
    return true;
}

// 실행
const analysisResult = analyzeGameStructure();

// 결과를 전역 변수에 저장
window.gameAnalysisResult = analysisResult;

console.log('\n📋 요약:');
console.log(`- Stage 발견: ${analysisResult.stage ? '✅' : '❌'}`);
console.log(`- ExportRoot 발견: ${analysisResult.exportRoot ? '✅' : '❌'}`);
console.log(`- 게임 보드 후보: ${analysisResult.potentialBoards.length}개`);
console.log(`- Canvas 개수: ${analysisResult.canvasInfo?.length || 0}개`);

if (analysisResult.potentialBoards.length > 0) {
    console.log('\n🎯 가장 유력한 게임 보드 후보:');
    const best = analysisResult.potentialBoards[0];
    console.log(`${best.path} - 크기: ${best.size}`);
    console.log('다음 명령어로 확인: window.gameAnalysisResult.potentialBoards[0].data');
}