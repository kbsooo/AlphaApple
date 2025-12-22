// FruitBox Solver Content Script

let session = null;
let rects = [];
const ROWS = 10;
const COLS = 17;

// Ensure ORT loads WASM from extension bundle (no threads for compatibility)
if (typeof ort !== "undefined" && ort.env && ort.env.wasm) {
    ort.env.wasm.wasmPaths = chrome.runtime.getURL("");
    ort.env.wasm.numThreads = 1;
}

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

// Fetch board from page context via background script (MAIN world)
function getBoardState() {
    return new Promise((resolve) => {
        chrome.runtime.sendMessage({ action: "getBoard" }, (response) => {
            if (chrome.runtime.lastError) {
                console.error("AlphaApple: getBoard failed", chrome.runtime.lastError);
                resolve(null);
                return;
            }
            resolve(response && response.board ? response.board : null);
        });
    });
}

async function getBoardStateWithRetry(maxAttempts = 8, delayMs = 200) {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const board = await getBoardState();
        if (board && board.length === 170) {
            return board;
        }
        await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    return null;
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
async function showOverlay(rectIdx) {
    const coords = await new Promise((resolve) => {
        chrome.runtime.sendMessage(
            { action: "getCoords", rect: rects[rectIdx] },
            (response) => {
                if (chrome.runtime.lastError) {
                    console.error("AlphaApple: getCoords failed", chrome.runtime.lastError);
                    resolve(null);
                    return;
                }
                resolve(response && response.coords ? response.coords : null);
            }
        );
    });
    if (!coords) {
        console.log("AlphaApple: Could not read coords from page.");
        return;
    }
    const { x1, y1, x2, y2, cellW, cellH } = coords;
    renderBox(x1, y1, x2, y2, cellW, cellH);
}

function renderBox(x1, y1, x2, y2, cellW, cellH) {
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
    const padX = (typeof cellW === "number" && cellW > 0) ? cellW / 2 : 20;
    const padY = (typeof cellH === "number" && cellH > 0) ? cellH / 2 : 20;
    overlay.style.left = (x1 - padX) + 'px';
    overlay.style.top = (y1 - padY) + 'px';
    overlay.style.width = (x2 - x1 + padX * 2) + 'px';
    overlay.style.height = (y2 - y1 + padY * 2) + 'px';
}

// Main Solver Loop
async function solve() {
    if (!session) await initSession();
    if (!session) return;

    const board = await getBoardStateWithRetry();
    if (!board || board.length !== 170) {
        console.log("AlphaApple: Failed to read board.");
        return;
    }
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
