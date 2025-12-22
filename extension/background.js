chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    const tabId = sender.tab && sender.tab.id;
    if (!tabId) {
        sendResponse({ error: "No active tab." });
        return;
    }

    if (request.action === "getBoard") {
        chrome.scripting.executeScript({
            target: { tabId },
            world: "MAIN",
            func: () => {
                const extractValue = (a) => {
                    if (!a) return 0;
                    if (typeof a.nu === "number") return a.nu;
                    if (typeof a.num === "number") return a.num;
                    if (typeof a.n === "number") return a.n;
                    if (Array.isArray(a.children)) {
                        for (const child of a.children) {
                            if (typeof child.text === "string") {
                                const v = parseInt(child.text, 10);
                                if (!Number.isNaN(v)) return v;
                            }
                        }
                    }
                    return 0;
                };

                const clusterCenters = (values, expectedCount) => {
                    if (!values.length) return [];
                    const sorted = values.slice().sort((a, b) => a - b);
                    const deltas = [];
                    for (let i = 1; i < sorted.length; i++) {
                        const d = sorted[i] - sorted[i - 1];
                        if (d > 0) deltas.push(d);
                    }
                    const median = deltas.length
                        ? deltas.sort((a, b) => a - b)[Math.floor(deltas.length / 2)]
                        : 1;
                    const threshold = Math.max(1, median / 2);

                    const clusters = [[sorted[0]]];
                    for (let i = 1; i < sorted.length; i++) {
                        const last = clusters[clusters.length - 1];
                        if (sorted[i] - last[last.length - 1] <= threshold) {
                            last.push(sorted[i]);
                        } else {
                            clusters.push([sorted[i]]);
                        }
                    }
                    let centers = clusters.map((c) => c.reduce((s, v) => s + v, 0) / c.length);
                    if (centers.length !== expectedCount) {
                        const min = sorted[0];
                        const max = sorted[sorted.length - 1];
                        const step = expectedCount > 1 ? (max - min) / (expectedCount - 1) : 1;
                        centers = Array.from({ length: expectedCount }, (_, i) => min + i * step);
                    }
                    return centers;
                };

                const nearestIndex = (centers, value) => {
                    let best = 0;
                    let bestDist = Infinity;
                    for (let i = 0; i < centers.length; i++) {
                        const d = Math.abs(centers[i] - value);
                        if (d < bestDist) {
                            bestDist = d;
                            best = i;
                        }
                    }
                    return best;
                };

                if (!window.exportRoot || !window.exportRoot.mm || !window.exportRoot.mm.mg) {
                    return null;
                }
                const apples = window.exportRoot.mm.mg.children;
                if (!apples || apples.length < 170) {
                    return null;
                }

                const xs = apples.map((a) => Math.round(a.x));
                const ys = apples.map((a) => Math.round(a.y));
                const xCenters = clusterCenters(xs, 17);
                const yCenters = clusterCenters(ys, 10);
                if (xCenters.length !== 17 || yCenters.length !== 10) {
                    return null;
                }

                const grid = Array.from({ length: 10 }, () => Array(17).fill(null));
                for (const a of apples) {
                    const r = nearestIndex(yCenters, a.y);
                    const c = nearestIndex(xCenters, a.x);
                    if (r >= 0 && r < 10 && c >= 0 && c < 17) {
                        grid[r][c] = a;
                    }
                }

                const board = [];
                for (let r = 0; r < 10; r++) {
                    for (let c = 0; c < 17; c++) {
                        const a = grid[r][c];
                        board.push(a && a.visible ? extractValue(a) : 0);
                    }
                }
                return board;
            },
        }).then((results) => {
            const board = results && results[0] ? results[0].result : null;
            sendResponse({ board });
        }).catch((err) => {
            sendResponse({ error: String(err) });
        });
        return true;
    }

    if (request.action === "getCoords") {
        chrome.scripting.executeScript({
            target: { tabId },
            world: "MAIN",
            func: (rect) => {
                const clusterCenters = (values, expectedCount) => {
                    if (!values.length) return [];
                    const sorted = values.slice().sort((a, b) => a - b);
                    const deltas = [];
                    for (let i = 1; i < sorted.length; i++) {
                        const d = sorted[i] - sorted[i - 1];
                        if (d > 0) deltas.push(d);
                    }
                    const median = deltas.length
                        ? deltas.sort((a, b) => a - b)[Math.floor(deltas.length / 2)]
                        : 1;
                    const threshold = Math.max(1, median / 2);

                    const clusters = [[sorted[0]]];
                    for (let i = 1; i < sorted.length; i++) {
                        const last = clusters[clusters.length - 1];
                        if (sorted[i] - last[last.length - 1] <= threshold) {
                            last.push(sorted[i]);
                        } else {
                            clusters.push([sorted[i]]);
                        }
                    }
                    let centers = clusters.map((c) => c.reduce((s, v) => s + v, 0) / c.length);
                    if (centers.length !== expectedCount) {
                        const min = sorted[0];
                        const max = sorted[sorted.length - 1];
                        const step = expectedCount > 1 ? (max - min) / (expectedCount - 1) : 1;
                        centers = Array.from({ length: expectedCount }, (_, i) => min + i * step);
                    }
                    return centers;
                };

                const medianDelta = (centers) => {
                    if (centers.length < 2) return null;
                    const diffs = [];
                    for (let i = 1; i < centers.length; i++) {
                        const d = centers[i] - centers[i - 1];
                        if (d > 0) diffs.push(d);
                    }
                    if (!diffs.length) return null;
                    diffs.sort((a, b) => a - b);
                    return diffs[Math.floor(diffs.length / 2)];
                };

                const nearestIndex = (centers, value) => {
                    let best = 0;
                    let bestDist = Infinity;
                    for (let i = 0; i < centers.length; i++) {
                        const d = Math.abs(centers[i] - value);
                        if (d < bestDist) {
                            bestDist = d;
                            best = i;
                        }
                    }
                    return best;
                };

                if (!window.exportRoot || !window.exportRoot.mm || !window.exportRoot.mm.mg) {
                    return null;
                }
                const apples = window.exportRoot.mm.mg.children;
                if (!apples || apples.length < 170) {
                    return null;
                }

                const xs = apples.map((a) => Math.round(a.x));
                const ys = apples.map((a) => Math.round(a.y));
                const xCenters = clusterCenters(xs, 17);
                const yCenters = clusterCenters(ys, 10);
                if (xCenters.length !== 17 || yCenters.length !== 10) {
                    return null;
                }

                const grid = Array.from({ length: 10 }, () => Array(17).fill(null));
                for (const a of apples) {
                    const r = nearestIndex(yCenters, a.y);
                    const c = nearestIndex(xCenters, a.x);
                    if (r >= 0 && r < 10 && c >= 0 && c < 17) {
                        grid[r][c] = a;
                    }
                }

                const [r1, c1, r2, c2] = rect;
                const a1 = grid[r1][c1];
                const a2 = grid[r2][c2];
                const x1 = a1 ? a1.x : xCenters[c1];
                const y1 = a1 ? a1.y : yCenters[r1];
                const x2 = a2 ? a2.x : xCenters[c2];
                const y2 = a2 ? a2.y : yCenters[r2];
                const cellW = medianDelta(xCenters);
                const cellH = medianDelta(yCenters);
                return { x1, y1, x2, y2, cellW, cellH };
            },
            args: [request.rect],
        }).then((results) => {
            const coords = results && results[0] ? results[0].result : null;
            sendResponse({ coords });
        }).catch((err) => {
            sendResponse({ error: String(err) });
        });
        return true;
    }
});
