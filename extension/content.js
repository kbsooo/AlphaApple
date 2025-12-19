// FruitBox Solver Content Script

let session = null;
let rects = [];
const ROWS = 10;
const COLS = 17;

// Generate rectangles (must match Python implementation)
function generateRects() {
    const list = [];
    for (let r1 = 0; r1 < ROWS; r1++) {
        for (let r2 = r1; r2 < ROWS; r2++) {
            for (let c1 = 0; c1 < COLS; c1++) {
                for (let c2 = c1; c2 < COLS; c2++) {
                    list.push([r1, c1, r2, c2]);
                }
            }
        }
    }
    return list;
}

rects = generateRects();

// Initialize ONNX Session
async function initSession() {
    const modelUrl = chrome.runtime.getURL('model.onnx');
    try {
        session = await ort.InferenceSession.create(modelUrl);
        console.log("AlphaApple: ONNX model loaded.");
    } catch (e) {
        console.error("AlphaApple: Failed to load model", e);
    }
}

// Scrape board from CreateJS state
function getBoardState() {
    // We need to execute this in the page context to access global vars
    // Content scripts run in isolated worlds.
    // We can use a custom event or a script injection.
    return new Promise((resolve) => {
        window.addEventListener('message', function handler(event) {
            if (event.data.type === 'ALPHAPPLE_BOARD_DATA') {
                window.removeEventListener('message', handler);
                resolve(event.data.board);
            }
        });

        const script = document.createElement('script');
        script.textContent = `
            (function() {
                if (window.exportRoot && window.exportRoot.mm && window.exportRoot.mm.mg) {
                    const apples = window.exportRoot.mm.mg.children;
                    const board = [];
                    for (let i = 0; i < 170; i++) {
                        const a = apples[i];
                        board.push(a.visible ? a.nu : 0);
                    }
                    window.postMessage({ type: 'ALPHAPPLE_BOARD_DATA', board: board }, '*');
                }
            })();
        `;
        document.head.appendChild(script);
        script.remove();
    });
}

// One-hot encoding for the model
function preprocess(boardArray) {
    // Input should be (1, 10, 10, 17)
    const float32Data = new Float32Array(1 * 10 * 10 * 17);
    for (let i = 0; i < 170; i++) {
        const val = boardArray[i];
        if (val >= 0 && val <= 9) {
            const r = Math.floor(i / 17);
            const c = i % 17;
            // index = channel * (rows * cols) + row * cols + col
            const index = val * (10 * 17) + r * 17 + c;
            float32Data[index] = 1.0;
        }
    }
    return new ort.Tensor('float32', float32Data, [1, 10, 10, 17]);
}

// Action masking logic
function getActionMask(boardArray) {
    const mask = new Uint8Array(rects.length);
    // Create 2D board for easy summing
    const board2D = [];
    for (let i = 0; i < ROWS; i++) {
        board2D.push(boardArray.slice(i * COLS, (i + 1) * COLS));
    }

    // Prefix sum for fast rectangle sum
    const ps = Array.from({ length: ROWS + 1 }, () => new Int32Array(COLS + 1));
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            ps[r + 1][c + 1] = board2D[r][c] + ps[r][c + 1] + ps[r + 1][c] - ps[r][c];
        }
    }

    for (let i = 0; i < rects.length; i++) {
        const [r1, c1, r2, c2] = rects[i];
        const sum = ps[r2 + 1][c2 + 1] - ps[r1][c2 + 1] - ps[r2 + 1][c1] + ps[r1][c1];
        if (sum === 10) {
            mask[i] = 1;
        }
    }
    return mask;
}

// Draw overlay
function showOverlay(rectIdx) {
    const [r1, c1, r2, c2] = rects[rectIdx];

    // We need to find the pixel coordinates.
    // The apples in JS have x,y. We can fetch them.
    const script = document.createElement('script');
    script.textContent = `
        (function() {
            const apples = window.exportRoot.mm.mg.children;
            const a1 = apples[${r1 * 17 + c1}];
            const a2 = apples[${r2 * 17 + c2}];
            // Send coordinates back
            window.postMessage({ 
                type: 'ALPHAPPLE_COORD_DATA', 
                coords: { x1: a1.x, y1: a1.y, x2: a2.x, y2: a2.y } 
            }, '*');
        })();
    `;

    window.addEventListener('message', function handler(event) {
        if (event.data.type === 'ALPHAPPLE_COORD_DATA') {
            window.removeEventListener('message', handler);
            const { x1, y1, x2, y2 } = event.data.coords;
            renderBox(x1, y1, x2, y2);
        }
    });
    document.head.appendChild(script);
    script.remove();
}

function renderBox(x1, y1, x2, y2) {
    let overlay = document.getElementById('alphapple-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'alphapple-overlay';
        overlay.style.position = 'absolute';
        overlay.style.border = '4px solid red';
        overlay.style.pointerEvents = 'none';
        overlay.style.zIndex = '9999';
        overlay.style.borderRadius = '10px';
        overlay.style.boxShadow = '0 0 10px rgba(255,0,0,0.5)';
        document.getElementById('animation_container').appendChild(overlay);
    }

    // Map Canvas coords to Page coords if necessary
    // Assuming canvas is top-left in container
    const PADDING = 20; // Approx apple radius
    overlay.style.left = (x1 - PADDING) + 'px';
    overlay.style.top = (y1 - PADDING) + 'px';
    overlay.style.width = (x2 - x1 + PADDING * 2) + 'px';
    overlay.style.height = (y2 - y1 + PADDING * 2) + 'px';
}

// Main Solver Loop
async function solve() {
    if (!session) await initSession();
    if (!session) return;

    const board = await getBoardState();
    const inputTensor = preprocess(board);
    const mask = getActionMask(board);

    const outputs = await session.run({ input: inputTensor });
    const qValues = outputs.output.data;

    let bestAction = -1;
    let maxQ = -Infinity;

    for (let i = 0; i < qValues.length; i++) {
        if (mask[i] && qValues[i] > maxQ) {
            maxQ = qValues[i];
            bestAction = i;
        }
    }

    if (bestAction !== -1) {
        console.log("AlphaApple: Best action found", rects[bestAction]);
        showOverlay(bestAction);
    } else {
        console.log("AlphaApple: No valid actions found.");
        const overlay = document.getElementById('alphapple-overlay');
        if (overlay) overlay.remove();
    }
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "solve") {
        solve();
    }
});
